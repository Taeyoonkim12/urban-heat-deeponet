"""Training and evaluation loops (identical protocol for every model)."""

import os
import gc
import time
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from .config import (CHUNK_SIZE, TOTAL_EPOCHS, LR_INIT, LR_MIN, WARMUP_EPOCHS,
                     PATIENCE, WEIGHT_DECAY, BETA_PHYS, PHYSICS_SEED)


def weighted_mse(pred, target, weights):
    return ((pred - target) ** 2 * weights.unsqueeze(1)).mean()


def make_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'training_log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    return log


def _unpack(batch, spec, device):
    batch = list(batch)
    branch = batch.pop(0).squeeze(0).to(device, non_blocking=True)
    trunk = batch.pop(0).squeeze(0).to(device, non_blocking=True)
    cat = batch.pop(0).squeeze(0).to(device, non_blocking=True) if spec['use_category'] else None
    target = batch.pop(0).squeeze(0).to(device, non_blocking=True)
    weights = batch.pop(0).squeeze(0).to(device, non_blocking=True)
    case_index = None
    if spec['use_seb']:
        case_index = int(batch.pop(0).item())
    if batch:
        raise ValueError(f'unexpected dataset fields: {len(batch)}')
    return branch, trunk, cat, target, weights, case_index


def _forward_chunked(model, branch, trunk, cat, chunk=CHUNK_SIZE):
    outs = []
    for i in range(0, trunk.shape[0], chunk):
        outs.append(model(branch[i:i + chunk], trunk[i:i + chunk],
                          None if cat is None else cat[i:i + chunk]))
    return torch.cat(outs, dim=0)


def train(model, spec, train_cases, test_cases, scalers, out_dir,
          dataset_cls, epochs=TOTAL_EPOCHS, patience=PATIENCE,
          schedule_epochs=None,
          physics_regularizer=None, physics_beta=BETA_PHYS,
          device=None, log=None, resume_from=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = log or make_logger(out_dir)
    use_amp = (device.type == 'cuda')
    schedule_epochs = int(epochs if schedule_epochs is None else schedule_epochs)
    if schedule_epochs < epochs:
        raise ValueError('schedule_epochs must be at least epochs')

    # Identical per-epoch test evaluation to every other model in the
    # ladder (random point subsample, as in the published runs).
    train_ds = dataset_cls(train_cases, scalers, spec)
    test_ds = dataset_cls(test_cases, scalers, spec)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    params = list(model.parameters())
    optimizer = optim.AdamW(params, lr=LR_INIT, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = ((epoch - WARMUP_EPOCHS) /
                    max(1, schedule_epochs - WARMUP_EPOCHS))
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return LR_MIN / LR_INIT + (1 - LR_MIN / LR_INIT) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_losses, test_losses, physics_losses = [], [], []
    physics_rng_state = None
    best = float('inf')
    pat_ctr = 0
    start_epoch = 0

    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        required = ('epoch', 'model_state_dict', 'optimizer_state_dict',
                    'scheduler_state_dict', 'amp_scaler_state_dict',
                    'train_losses', 'test_losses', 'physics_losses',
                    'physics_rng_state', 'numpy_rng_state',
                    'torch_rng_state', 'best_loss', 'patience_counter')
        missing = [name for name in required if name not in ck]
        if missing:
            raise ValueError(
                f'resume checkpoint lacks exact-continuation state: {missing}')
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        amp_scaler.load_state_dict(ck['amp_scaler_state_dict'])
        train_losses = list(ck.get('train_losses', []))
        test_losses = list(ck.get('test_losses', []))
        physics_losses = list(ck.get('physics_losses', []))
        physics_rng_state = ck.get('physics_rng_state')
        start_epoch = int(ck['epoch'])
        best = float(ck['best_loss'])
        pat_ctr = int(ck['patience_counter'])
        np.random.set_state(ck['numpy_rng_state'])
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        if torch.cuda.is_available() and ck.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in ck['cuda_rng_state_all']])
        if start_epoch > epochs:
            raise ValueError(
                f'resume checkpoint epoch {start_epoch} exceeds --epochs {epochs}')
        if pat_ctr >= patience and start_epoch < epochs:
            raise ValueError(
                'cannot resume a run that already met its early-stopping gate')
        log(f"RESUMED from {resume_from}: epoch {start_epoch}, "
            f"best test {best:.6f} — continuing to {epochs}")

    log(f"training: epochs={epochs}, schedule_epochs={schedule_epochs}, "
        f"lr={LR_INIT}->{LR_MIN}, patience={patience}, "
        f"params={sum(p.numel() for p in model.parameters()):,}")

    # Save scalers up front so an interrupted run keeps everything needed
    # to resume/evaluate.
    with open(os.path.join(out_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    t0 = time.time()
    physics_rng = np.random.default_rng(PHYSICS_SEED)
    if physics_rng_state is not None:
        physics_rng.bit_generator.state = physics_rng_state

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, epoch_phys, n_batch = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"E{epoch+1}/{epochs}", leave=False):
            branch, trunk, cat, target, weights, case_index = _unpack(
                batch, spec, device)
            optimizer.zero_grad(set_to_none=True)
            n_pts = trunk.shape[0]
            n_chunks = max(1, (n_pts + CHUNK_SIZE - 1) // CHUNK_SIZE)
            batch_loss = 0.0
            for ci in range(n_chunks):
                s, e = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, n_pts)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(branch[s:e], trunk[s:e],
                                 None if cat is None else cat[s:e])
                    # Per-chunk mean loss, identical to the published training
                    # protocol of M0-M4 (each chunk backward once).
                    loss = weighted_mse(pred, target[s:e], weights[s:e])
                amp_scaler.scale(loss).backward()
                batch_loss += loss.item()

            if physics_regularizer is not None:
                if case_index is None:
                    raise RuntimeError('physics model batch has no case index')
                case = train_ds.cases[case_index]
                phys_loss, phys_info = physics_regularizer(
                    model, case['params'], physics_rng)
                weighted_phys = float(physics_beta) * phys_loss
                amp_scaler.scale(weighted_phys).backward()
                batch_loss += weighted_phys.item()
                epoch_phys += phys_info['loss']
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += batch_loss / n_chunks
            n_batch += 1

        train_loss = epoch_loss / max(n_batch, 1)
        physics_loss = epoch_phys / max(n_batch, 1)
        physics_losses.append(physics_loss)
        scheduler.step()

        model.eval()
        test_loss, tn = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                branch, trunk, cat, target, weights = _unpack(batch, spec, device)[:5]
                pred = _forward_chunked(model, branch, trunk, cat)
                test_loss += weighted_mse(pred, target, weights).item()
                tn += 1
        test_loss /= max(tn, 1)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        improved = test_loss < best
        if improved:
            best = test_loss
            pat_ctr = 0
        else:
            pat_ctr += 1

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'amp_scaler_state_dict': amp_scaler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_loss': best,
            'patience_counter': pat_ctr,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'physics_losses': physics_losses,
            'physics_rng_state': physics_rng.bit_generator.state,
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': (torch.cuda.get_rng_state_all()
                                   if torch.cuda.is_available() else None),
        }
        if improved:
            torch.save(ckpt, os.path.join(out_dir, 'best_model.pt'))
        torch.save(ckpt, os.path.join(out_dir, 'last_model.pt'))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            phys_txt = (f" | physics {physics_loss:.6f}"
                        if physics_regularizer is not None else '')
            log(f"  epoch {epoch+1:>4d} | train {train_loss:.6f} | test {test_loss:.6f}"
                f"{phys_txt} | best {best:.6f} | P {pat_ctr}/{patience} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}")
        if pat_ctr >= patience:
            log(f"early stop at epoch {epoch+1}")
            break
        if (epoch + 1) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log(f"done in {(time.time()-t0)/3600:.2f} h, best test loss {best:.6f}")
    with open(os.path.join(out_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'physics_losses': physics_losses,
                     'best_test_loss': best}, f)
    with open(os.path.join(out_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    return best


def evaluate(model, spec, test_cases, scalers, out_dir, dataset_cls,
             model_name, device=None, log=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = log or make_logger(out_dir)

    ckpt_path = os.path.join(out_dir, 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log(f"loaded best checkpoint (epoch {ckpt.get('epoch', '?')})")

    test_ds = dataset_cls(test_cases, scalers, spec)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluate"):
            branch, trunk, cat, target = _unpack(batch, spec, device)[:4]
            pred = _forward_chunked(model, branch, trunk, cat)
            preds.append(pred.float().cpu().numpy())
            targets.append(target.cpu().numpy())

    preds = scalers['output'].inverse_transform(np.concatenate(preds))
    targets = scalers['output'].inverse_transform(np.concatenate(targets))

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    log(f"[{model_name}] test MAE {mae:.4f} C | RMSE {rmse:.4f} C | R2 {r2:.6f} "
        f"| {len(targets):,} points")

    results = {'model_name': model_name, 'mae': mae, 'rmse': rmse, 'r2': r2,
               'total_params': sum(p.numel() for p in model.parameters())}
    with open(os.path.join(out_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return results
