# In[1]: load the dataset
import numpy as np
from datasets import load_dataset
repo_id = "kshitij-pandey/navier_stokes_datasets"
params = load_dataset(repo_id, "params", split="train")
snapshots = load_dataset(repo_id, "snapshots_split", split="train")
triang_data = load_dataset(repo_id, "triangles", split="train")
coords_data = load_dataset(repo_id, "coords", split="train")
print(snapshots.column_names)
# In[2]:
for name in list(snapshots.column_names):
    print(f'Shape of {name:2s} snapshots matrix: {np.array(snapshots[name]).shape}')
print(f'Shape of parameters matrix: {params.shape}')

# In[3]:
import matplotlib.tri as mtri

# Extract coordinates: coords_data[0] for x, coords_data[1] for y
x = np.array([coords_data[0][f'col_{i}'] for i in range(len(coords_data[0]))])
y = np.array([coords_data[1][f'col_{i}'] for i in range(len(coords_data[1]))])
xy = np.stack([x, y], axis=1)  # shape: (1639, 2)

triang_conn = np.array([[row['col_0'], row['col_1'], row['col_2']] for row in triang_data])

triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triang_conn)

# In[4]:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 8), sharey=True, sharex=True)
ax = ax.flatten()
for i in range(9):
    ax[i].tricontourf(triang, np.array(snapshots['vx'])[i], levels=16)
    inlet_velocity = params[i]['col_0']
    ax[i].set_title(f'Original snapshot at inlet velocity = {round(inlet_velocity, 2)}')
plt.show()

# In[5]:
param_values = np.array([entry['col_0'] for entry in params])
snapshots_vx = np.array(snapshots['vx'])

from ezyrb import Database, POD, RBF, ReducedOrderModel as ROM

db = Database(param_values.reshape(-1, 1), snapshots_vx)  # reshape needed here

rom = ROM(db, POD(), RBF())
rom.fit()

# In[6]:
new_params = np.random.uniform(size=(2)) * 79. + 1.

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 3))
for i, param in enumerate(new_params):
    ax[i].tricontourf(triang, *rom.predict([param]).snapshots_matrix)
    ax[i].set_title(f'Predicted snapshots at inlet velocity = {param}')
plt.show()

# In[7]:
errors = rom.kfold_cv_error(n_splits=5)
print('Average error for each fold:')
for e in errors:
    print('  ', e)
print('\nAverage error =', errors.mean())

# In[9]:
from ezyrb import AE, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, ANN
import torch.nn as nn
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

# In[10]:
reductions['AE'] = AE([100, 10], [10, 100], nn.ReLU(), nn.ReLU(), 30000)
approximations['ANN'] = ANN([50, 10], nn.ReLU(), 30000)
