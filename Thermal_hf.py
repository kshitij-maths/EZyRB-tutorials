# --- Load necessary libraries ---

from datasets import load_dataset
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

from ezyrb import Database, POD, AE, RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, ANN, ReducedOrderModel as ROM

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", message="Ill-conditioned matrix ")

# --- Load data from Hugging Face Hub ---

print("Loading datasets from Hugging Face Hub...")

params_ds = load_dataset("kshitij-pandey/termal_dataset", "params", split="train")
snapshots_ds = load_dataset("kshitij-pandey/termal_dataset", "snapshots", split="train")
triangles_ds = load_dataset("kshitij-pandey/termal_dataset", "triangles", split="train")
coords_ds = load_dataset("kshitij-pandey/termal_dataset", "coords", split="train")

# print("Datasets loaded.")
# print("params_ds columns:", params_ds.column_names)
# print("snapshots_ds columns (first 5):", snapshots_ds.column_names[:5])
# print("triangles_ds columns:", triangles_ds.column_names)
# print("coords_ds columns:", coords_ds.column_names)

# --- Auto-detect columns and convert datasets to numpy arrays ---

def dataset_to_array(ds, dtype=float):
    cols = ds.column_names
    return np.array([[entry[col] for col in cols] for entry in ds], dtype=dtype)

params = dataset_to_array(params_ds)
snapshots_raw = dataset_to_array(snapshots_ds)
triangles = dataset_to_array(triangles_ds, dtype=int)
coords = dataset_to_array(coords_ds).T

# Confirm shapes
print(f"params shape: {params.shape}")
print(f"snapshots_raw shape: {snapshots_raw.shape}")
print(f"triangles shape: {triangles.shape}, max triangle index: {triangles.max()}")
print(f"coords shape: {coords.shape}, number of points: {coords.shape[0]}")

# --- Check triangle indices vs coords
if triangles.max() >= coords.shape[0]:
    raise ValueError(f"Invalid triangulation: triangles refer to point {triangles.max()}, but only {coords.shape[0]} coordinates exist.")

# --- Format data for ezyrb
snapshots = {'vx': snapshots_raw.T}
triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

class Data:
    def __init__(self, params, snapshots, triang):
        self.params = params
        self.snapshots = snapshots
        self.triang = triang

data = Data(params, snapshots, triang)

# --- Check loaded data
print(f"Shape of vx snapshots: {data.snapshots['vx'].shape}")
print(f"Shape of params: {data.params.shape}")
print("coords shape:", coords.shape)
print("triangles shape:", triangles.shape)
print("one snapshot shape:", data.snapshots['vx'][0].shape)
print("number of triang points:", len(data.triang.x))

# --- Plot original snapshots
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 8), sharey=True, sharex=True)
ax = ax.flatten()
for i in range(8):
    ax[i].tricontourf(data.triang, data.snapshots['vx'][:,i], levels=16)
    ax[i].set_title(f'Original snapshot at inlet velocity = {data.params[i].round(2)}')
plt.tight_layout()
plt.show()

# --- Build and fit reduced order model
db = Database(data.params, data.snapshots['vx'].T)
rom = ROM(db, POD(), RBF())
rom.fit()

# --- Predict new parameters
new_params = np.random.uniform(size=(2, 2)) * 79. + 1.  # 2 parameters, each with 2 values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 3))

for i, param in enumerate(new_params):
    predicted_snap = rom.predict(param.reshape(1, -1)).snapshots_matrix.flatten()  # shape (1,2)
    ax[i].tricontourf(data.triang, predicted_snap)
    ax[i].set_title(f'Predicted snapshots at inlet velocity = {param.round(2)}')

plt.tight_layout()
plt.show()

# --- Cross-validation error
errors = rom.kfold_cv_error(n_splits=5)
print('Average error for each fold:')
for e in errors:
    print(f'  {e}')
print(f'\nAverage error = {errors.mean()}')

# --- Try various reduction and approximation methods
reductions = {
    'POD': POD('svd', rank=10),
    'AE': AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 10),
}

approximations = {
    'RBF': RBF(),
    'GPR': GPR(),
    'KNeighbors': KNeighborsRegressor(),
    'RadiusNeighbors': RadiusNeighborsRegressor(),
    'ANN': ANN([20, 20], nn.Tanh(), 10),
}

header = '{:10s}'.format('')
for name in approximations:
    header += f' {name:>15s}'
print(header)

for redname, redclass in reductions.items():
    row = f'{redname:10s}'
    for approxname, approxclass in approximations.items():
        rom = ROM(db, redclass, approxclass)
        rom.fit()
        row += f' {rom.kfold_cv_error(n_splits=5).mean():15e}'
    print(row)

# Alternate config for AE & ANN
reductions['AE'] = AE([100, 10], [10, 100], nn.ReLU(), nn.ReLU(), 30000)
approximations['ANN'] = ANN([50, 10], nn.ReLU(), 30000)
