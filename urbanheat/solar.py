"""Solar geometry, radiative surface properties and shadow computation
used by the physics-consistent models (M5 family)."""

import numpy as np

# Hourly solar position and irradiance for the simulated day
# (columns: hour, azimuth [rad], altitude [rad],
#  BHI = horizontal direct irradiance [W/m2], DHI = diffuse [W/m2]).
# NOTE: column 3 is the direct irradiance on a HORIZONTAL plane
# (BHI = DNI * sin(altitude)); it is NOT the direct normal irradiance.
# Verified numerically: BHI / sin(altitude) is ~461-469 W/m2 for all hours.
# Values match the CFD solar loading.
SOLAR_TABLE = np.array([
    [7,  1.40, 0.25, 114.07,  48.89],
    [8,  1.55, 0.45, 203.92,  87.39],
    [9,  1.72, 0.66, 285.44, 122.33],
    [10, 1.93, 0.86, 352.97, 151.27],
    [11, 2.32, 1.04, 401.87, 172.23],
    [12, 2.72, 1.17, 428.79, 183.77],
    [13, 3.40, 1.19, 431.88, 185.09],
    [14, 3.95, 1.08, 410.94, 176.12],
    [15, 4.29, 0.91, 367.38, 157.45],
    [16, 4.51, 0.71, 304.17, 130.36],
    [17, 4.69, 0.51, 225.64,  96.70],
    [18, 4.84, 0.30, 137.20,  58.80],
], dtype=np.float64)

HOUR_TO_ROW = {int(r[0]): i for i, r in enumerate(SOLAR_TABLE)}

_az, _alt = SOLAR_TABLE[:, 1], SOLAR_TABLE[:, 2]
SUN_VECS = np.column_stack([np.sin(_az) * np.cos(_alt),
                            np.cos(_az) * np.cos(_alt),
                            np.sin(_alt)])
SUN_BHI = SOLAR_TABLE[:, 3].astype(np.float32)
SUN_DIF = SOLAR_TABLE[:, 4].astype(np.float32)

# Direct normal irradiance recovered from the horizontal component.
# A sin(altitude) floor guards the low-sun hours.
_SIN_ALT = np.sin(SOLAR_TABLE[:, 2]).astype(np.float32)
SUN_DNI = np.where(_SIN_ALT > 0.05, SUN_BHI / np.maximum(_SIN_ALT, 0.05),
                   0.0).astype(np.float32)

# Radiative properties per surface category [Building, Road, Green, Water,
# Topo_IN], matching the CFD boundary-condition inputs (paper Table 2).
EPS_CAT = np.array([0.50, 0.95, 0.35, 0.80, 0.70], dtype=np.float32)  # emissivity
RHO_CAT = np.array([0.50, 0.05, 0.65, 0.10, 0.30], dtype=np.float32)  # reflectivity
TAU_CAT = np.array([0.00, 0.00, 0.00, 0.10, 0.00], dtype=np.float32)  # transmissivity


def solar_branch(params):
    """Branch input for the solar-branch models:
    [RH, T_amb, altitude (rad), azimuth (rad), BHI (W/m2)].
    The horizontal direct irradiance (BHI) is kept as the input covariate
    (it varies with hour, unlike the nearly constant DNI); only the SEB
    flux computation below uses the recovered DNI."""
    row = SOLAR_TABLE[HOUR_TO_ROW[int(round(params['hour']))]]
    return np.array([params['humidity'], params['temperature'],
                     row[2], row[1], row[3]], dtype=np.float32)


def compute_shadow_masks(mesh, ref_points, ref_normals, chunk=200_000,
                         cache_path=None, log=print):
    """Sun-visibility masks (12 hours x N reference points) by ray casting
    against the building STL. Cached on disk because this is the slowest
    preprocessing step."""
    import os
    n = len(ref_points)
    if cache_path is not None and os.path.exists(cache_path):
        lit = np.load(cache_path)
        assert lit.shape == (12, n)
        return lit

    ray_ok = True
    try:
        mesh.ray.intersects_any(ray_origins=np.array([[0.0, 0.0, 1e6]]),
                                ray_directions=np.array([[0.0, 0.0, 1.0]]))
    except BaseException:
        ray_ok = False
        log("ray casting unavailable, falling back to orientation-only shading")

    if not ray_ok:
        lit = np.zeros((12, n), dtype=bool)
        for r in range(12):
            lit[r] = (ref_normals @ SUN_VECS[r]) > 0
        return lit

    lit = np.ones((12, n), dtype=bool)
    for r in range(12):
        sv = SUN_VECS[r]
        facing = (ref_normals @ sv) > 0
        lit[r, ~facing] = False
        idx = np.where(facing)[0]
        for c0 in range(0, len(idx), chunk):
            ii = idx[c0:c0 + chunk]
            hit = mesh.ray.intersects_any(
                ray_origins=ref_points[ii] + ref_normals[ii] * 0.5 + sv * 0.5,
                ray_directions=np.tile(sv, (len(ii), 1)))
            lit[r, ii] = ~hit
        log(f"  shadow hr={int(SOLAR_TABLE[r, 0]):02d}: lit {int(lit[r].sum()):,}/{n:,}")

    if cache_path is not None:
        np.save(cache_path, lit)
    return lit


def compute_qsw(case, ref_tree, lit_ref):
    """Absorbed shortwave flux per point for one case. Shadow state is
    inherited from the nearest reference point."""
    r = HOUR_TO_ROW[int(round(case['params']['hour']))]
    sv = SUN_VECS[r]
    nrm = case['normal'].astype(np.float64)
    cosi = np.maximum(0.0, nrm @ sv).astype(np.float32)
    skyf = ((1.0 + nrm[:, 2]) * 0.5).astype(np.float32)
    pts = np.column_stack([case['x_coords'], case['y_coords'], case['z_coords']]).astype(np.float64)
    _, j = ref_tree.query(pts, k=1, workers=-1)
    lit = lit_ref[r][j].astype(np.float32)
    rho = RHO_CAT[case['category'].astype(np.int64)]
    # Single projection: DNI (recovered from BHI) times the incidence
    # cosine. Using SUN_BHI here would double-apply the solar elevation.
    return ((1.0 - rho) * (SUN_DNI[r] * cosi * lit + SUN_DIF[r] * skyf)).astype(np.float32)


def _self_test():
    """Arithmetic consistency checks (run: python -m urbanheat.solar)."""
    alt = SOLAR_TABLE[:, 2]
    # 1) BHI == DNI * sin(altitude) after recovery
    assert np.allclose(SUN_DNI * np.sin(alt), SUN_BHI, rtol=1e-4), \
        "BHI reconstruction failed"
    # 2) recovered DNI is nearly constant across hours (clear-sky beam)
    assert SUN_DNI.max() - SUN_DNI.min() < 15.0, \
        f"DNI spread too large: {SUN_DNI.min():.1f}-{SUN_DNI.max():.1f}"
    # 3) a horizontal facet receives exactly BHI as its direct term
    horiz = np.array([0.0, 0.0, 1.0])
    for r in range(len(SOLAR_TABLE)):
        cosi = max(0.0, float(horiz @ SUN_VECS[r]))
        assert abs(SUN_DNI[r] * cosi - SUN_BHI[r]) < 1.0, f"hour row {r}"
    # 4) a facet facing away from the sun receives zero direct flux
    back = -SUN_VECS[6]
    assert max(0.0, float(back @ SUN_VECS[6])) == 0.0
    print("solar self-test OK | DNI ~", np.round(SUN_DNI, 1))


if __name__ == '__main__':
    _self_test()
