# Reference case

The published reference solve: 1000 W, 10 mTorr, 30% Ar, bias on at 200 W, under the production
closure. The runs behind figures 6, 9, 10, 11 and 12 come from here.

`summary.json` holds the scalars, including a `fix_provenance` block recording the exact closure
flags the run used. The `.npy` files are two-dimensional fields on the (r, z) mesh.

## Reading the field arrays

Every field array has shape **(50, 80) = (Nr, Nz)**: the first axis is radial, the second axial.
Values are SI, so densities are m^-3 and temperatures eV.

`mesh.npz` gives the coordinates, so no reconstruction from the solver is needed:

| key | shape | meaning |
|---|---|---|
| `r_centres_m` | (50,) | radial cell centres, metres, matching axis 0 |
| `z_centres_m` | (80,) | axial cell centres, metres, matching axis 1 |
| `r_faces_m` | (51,) | radial cell faces, for pcolormesh |
| `z_faces_m` | (81,) | axial cell faces |
| `cell_volume_m3` | (50, 80) | cell volumes, for volume-weighted averages |
| `inside_domain` | (50, 80) | boolean, true inside the plasma domain |

The domain spans r = 0 to 0.105 m and z = 0 to 0.202 m, the latter being the process chamber,
the aperture and the ICP source stacked together. The radial mesh is tanh-stretched toward the
wall (beta_r = 1.2); the axial mesh is uniform. Cells outside the reactor body carry zeros, so
mask with `inside_domain` before taking statistics. To
plot a field:

```python
import numpy as np, matplotlib.pyplot as plt
m  = np.load("mesh.npz")
nF = np.load("nF.npy")
plt.pcolormesh(m["z_faces_m"] * 100, m["r_faces_m"] * 100, nF, shading="flat")
plt.xlabel("z (cm)"); plt.ylabel("r (cm)")
```

The wafer plane is the first axial column, `nF[:, 0]`. That is the row figures 13 and 17 plot,
and `r_centres_m * 100` reproduces their radial axis in centimetres exactly.

Geometry constants are in `model/config/default_config.yaml` under `reactor_geometry`.
