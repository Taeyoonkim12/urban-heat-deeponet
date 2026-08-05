# Data layout

The CFD dataset used in the paper is not included in this repository
(see the data availability statement of the paper). The samples in
`data/dummy/` are synthetic - pure random numbers with the correct file
layout - intended for pipeline testing only; they are not physically
meaningful and reflect no statistics of the real data.

## Expected layout under `--data-root`

```
<data-root>/
  cases/                       CFD case exports (normal material)
    Case_{RH}_{T}_{Hr}_XYZInternalTable*.csv
  cases_climate/               climate-material scenario (M5-mat only)
  Building.csv                 per-category surface point exports
  Road.csv                     (used for d_BP and category labels)
  Green.csv
  Water.csv
  Topo_IN.csv
  building.stl                 building geometry (normals / shadows / SVF)
```

## Case CSV schema

Each case file contains one row per surface sample point:

| column | unit | description |
|---|---|---|
| `X (m)` | m | x coordinate |
| `Y (m)` | m | y coordinate |
| `Z (m)` | m | z coordinate |
| `Temperature (K)` | K | surface temperature |

The filename encodes the meteorological condition:
`Case_{RH}_{T}_{Hr}_...` with relative humidity RH (%), ambient
temperature T (deg C) and local solar hour Hr.

Category CSVs have columns `Point, X (m), Y (m), Z (m)` (the loader
reads columns 1-3 by position).
