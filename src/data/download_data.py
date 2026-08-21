import os
import subprocess
import zipfile
from pathlib import Path

def download_dataset():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Kaggle to use the local relative .kaggle directory/token
    import os
    import json
    project_root = Path(__file__).resolve().parents[2]
    local_kaggle = project_root / ".kaggle"
    
    # Read access_token and set env var if present (for new tokens)
    access_token_file = local_kaggle / "access_token"
    if access_token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = access_token_file.read_text().strip()
    
    # Read legacy kaggle.json and set env vars if present (for legacy credentials)
    kaggle_json_file = local_kaggle / "kaggle.json"
    if kaggle_json_file.exists():
        try:
            with open(kaggle_json_file) as f:
                creds = json.load(f)
                if "username" in creds:
                    os.environ["KAGGLE_USERNAME"] = creds["username"]
                if "key" in creds:
                    os.environ["KAGGLE_KEY"] = creds["key"]
        except Exception as e:
            print(f"Warning: Failed to read local kaggle.json: {e}")
    
    print("Downloading ieee-fraud-detection dataset via Kaggle API...")
    # Call the Kaggle CLI directly to download the competition files
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", "ieee-fraud-detection", "-p", str(raw_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error downloading dataset:")
        print(result.stderr)
        print("\nPlease make sure you have kaggle.json or access_token configured in your local .kaggle/ directory and have accepted the competition rules on Kaggle.")
        return
        
    print(result.stdout)
    
    print("Unzipping downloaded files...")
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        print("No zip files found to extract.")
        return

    for item in zip_files:
        print(f"Unzipping {item.name}...")
        with zipfile.ZipFile(item, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        item.unlink()  # Remove zip file to save space
        
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_dataset()
