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
                     PATIENCE, WEIGHT_DECAY, BETA_PHYS)


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
    qsw = tamb = None
    if spec['use_seb']:
        qsw = batch.pop(0).squeeze(0).to(device, non_blocking=True)
        tamb = batch.pop(0).to(device, non_blocking=True).float()
    return branch, trunk, cat, target, weights, qsw, tamb


def _forward_chunked(model, branch, trunk, cat, chunk=CHUNK_SIZE):
    outs = []
    for i in range(0, trunk.shape[0], chunk):
        outs.append(model(branch[i:i + chunk], trunk[i:i + chunk],
                          None if cat is None else cat[i:i + chunk]))
    return torch.cat(outs, dim=0)


def train(model, spec, train_cases, test_cases, scalers, out_dir,
          dataset_cls, epochs=TOTAL_EPOCHS, seb_loss=None, seb_head=None,
          device=None, log=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = log or make_logger(out_dir)
    use_amp = (device.type == 'cuda')

    train_ds = dataset_cls(train_cases, scalers, spec)
    test_ds = dataset_cls(test_cases, scalers, spec)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    params = list(model.parameters())
    if seb_head is not None:
        params += list(seb_head.parameters())
    optimizer = optim.AdamW(params, lr=LR_INIT, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return LR_MIN / LR_INIT + (1 - LR_MIN / LR_INIT) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    log(f"training: epochs={epochs}, lr={LR_INIT}->{LR_MIN}, patience={PATIENCE}, "
        f"params={sum(p.numel() for p in model.parameters()):,}")

    train_losses, test_losses = [], []
    best = float('inf')
    patience = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batch = 0.0, 0
        for batch in tqdm(train_loader, desc=f"E{epoch+1}/{epochs}", leave=False):
            branch, trunk, cat, target, weights, qsw, tamb = _unpack(batch, spec, device)
            optimizer.zero_grad(set_to_none=True)
            n_pts = trunk.shape[0]
            n_chunks = max(1, (n_pts + CHUNK_SIZE - 1) // CHUNK_SIZE)
            batch_loss = 0.0
            for ci in range(n_chunks):
                s, e = ci * CHUNK_SIZE, min((ci + 1) * CHUNK_SIZE, n_pts)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(branch[s:e], trunk[s:e],
                                 None if cat is None else cat[s:e])
                    loss = weighted_mse(pred, target[s:e], weights[s:e])
                if seb_loss is not None:
                    # SEB residual in fp32, outside autocast
                    loss = loss + BETA_PHYS * seb_loss(pred, qsw[s:e], cat[s:e], tamb)
                amp_scaler.scale(loss).backward()
                batch_loss += loss.item()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += batch_loss / n_chunks
            n_batch += 1

        train_loss = epoch_loss / max(n_batch, 1)
        scheduler.step()

        model.eval()
        test_loss, tn = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                branch, trunk, cat, target, weights, _, _ = _unpack(batch, spec, device)
                pred = _forward_chunked(model, branch, trunk, cat)
                test_loss += weighted_mse(pred, target, weights).item()
                tn += 1
        test_loss /= max(tn, 1)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best:
            best = test_loss
            patience = 0
            ckpt = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss, 'test_loss': test_loss,
                    'train_losses': train_losses, 'test_losses': test_losses}
            if seb_head is not None:
                ckpt['seb_head_state_dict'] = seb_head.state_dict()
            torch.save(ckpt, os.path.join(out_dir, 'best_model.pt'))
        else:
            patience += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log(f"  epoch {epoch+1:>4d} | train {train_loss:.6f} | test {test_loss:.6f} "
                f"| best {best:.6f} | P {patience}/{PATIENCE} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}")
        if patience >= PATIENCE:
            log(f"early stop at epoch {epoch+1}")
            break
        if (epoch + 1) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log(f"done in {(time.time()-t0)/3600:.2f} h, best test loss {best:.6f}")
    with open(os.path.join(out_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'test_losses': test_losses,
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
            branch, trunk, cat, target, _, _, _ = _unpack(batch, spec, device)
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
