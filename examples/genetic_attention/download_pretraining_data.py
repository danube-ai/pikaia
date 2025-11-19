# save as download_shards.py
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = "HuggingFaceTB/smollm-corpus"
folder = "fineweb-edu-dedup"
n_shards = 64  # change if you want more/less

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent
out_dir = script_dir / ".data"

for i in range(n_shards):
    fname = f"train-{i:05d}-of-00234.parquet"
    # Download directly to local_dir (skips cache copying)
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{folder}/{fname}",
        local_dir=out_dir,
    )
    print("Downloaded:", local_path)
