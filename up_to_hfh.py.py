import os
import numpy as np
import tempfile
import subprocess
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login, create_repo

# -------- USER CONFIGURATION --------
HF_TOKEN = "write your token here"
HF_REPO_ID = "kshitij-pandey/termal_dataset"  # replace your-username/directory(it will create a directory if it doesn't exists)
GITHUB_REPO_URL = "https://github.com/mathLab/Smithers.git"
TARGET_SUBDIR = "smithers/dataset/datasets/termal" #Replace termel with other directories (e.g. navier_stokes) to work with datasets in intended repositories
PRIVATE_REPO = False  # Set True if you want the repo private
# ------------------------------------

# Authenticate
login(token=HF_TOKEN)

def clone_and_find_npy(repo_url, subdir, tmp_dir):
    """Clone GitHub repo and return list of .npy file paths in the target subdirectory."""
    subprocess.run(["git", "clone", repo_url, tmp_dir], check=True)
    target_path = os.path.join(tmp_dir, subdir)
    return [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith(".npy")]

def npy_to_datasetdict(npy_path):
    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr = arr[:, None]
    columns = {f"col_{i}": arr[:, i].tolist() for i in range(arr.shape[1])}
    return DatasetDict({"train": Dataset.from_dict(columns)})

def main():
    api = HfApi()

    # Create the dataset repo if it doesn't exist
    existing = [d.id for d in api.list_datasets()]
    if HF_REPO_ID not in existing:
        print(f"Creating HF repo: {HF_REPO_ID}")
        create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=PRIVATE_REPO)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Cloning GitHub repo into: {tmp_dir}")
        npy_files = clone_and_find_npy(GITHUB_REPO_URL, TARGET_SUBDIR, tmp_dir)

        for npy_file in npy_files:
            config_name = Path(npy_file).stem
            print(f"Uploading config: {config_name}")
            dataset_dict = npy_to_datasetdict(npy_file)
            dataset_dict.push_to_hub(repo_id=HF_REPO_ID, config_name=config_name)

    print("DONE: All datasets pushed to the Hugging Face Hub.")

if __name__ == "__main__":
    main()
