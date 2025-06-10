import numpy as np
from datasets import load_dataset, Dataset
import getpass
from huggingface_hub import HfFolder

# Authenticate
token = getpass.getpass("Enter your token (input will not be visible): ")
HfFolder.save_token(token)

# Load dataset
print("Loading original snapshots from Hugging Face...")
snapshots_ds = load_dataset("kshitij-pandey/navier_stokes_datasets", "snapshots", split="train")

# Get all column names (should be col_0, col_1, ..., col_4916)
all_cols = snapshots_ds.column_names
print(f"Number of columns: {len(all_cols)}")

# Extract data: list of list, each inner list is snapshot vector for a sample
snap_array = np.array([[row[col] for col in all_cols] for row in snapshots_ds])

print(f"Snapshot shape: {snap_array.shape}")  # Should be (500, 4917)

# Number of DOFs per variable (vx, vy, p)
total_dofs = snap_array.shape[1]
dof = total_dofs // 3  # 1639

# Split data
vx = snap_array[:, 0:dof]
vy = snap_array[:, dof:2*dof]
p  = snap_array[:, 2*dof:]

# Compute magnitude
magv = np.sqrt(vx**2 + vy**2)

print(f"vx shape: {vx.shape}, vy shape: {vy.shape}, p shape: {p.shape}, mag(v) shape: {magv.shape}")

# Create new dataset dictionary with the desired keys and lists
new_data = {
    "vx": vx.tolist(),
    "vy": vy.tolist(),
    "mag(v)": magv.tolist(),
    "p": p.tolist(),
}

# Create Hugging Face Dataset from dictionary
dataset = Dataset.from_dict(new_data)

# Push dataset back to HF (use your repo and config)
dataset.push_to_hub("kshitij-pandey/navier_stokes_datasets", config_name="snapshots_split")

print("Dataset pushed to Hugging Face as 'snapshots_split'.")
