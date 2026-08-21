import os
import json
import zipfile
from pathlib import Path

def download_and_extract():
    project_dir = Path(__file__).resolve().parents[2]
    kaggle_dir = project_dir / ".kaggle"
    
    access_token_path = kaggle_dir / "access_token"
    kaggle_json_path = kaggle_dir / "kaggle.json"
    
    if access_token_path.exists():
        token = access_token_path.read_text().strip()
        os.environ["KAGGLE_API_TOKEN"] = token
        print("Set KAGGLE_API_TOKEN from .kaggle/access_token.")
    elif kaggle_json_path.exists():
        try:
            creds = json.loads(kaggle_json_path.read_text())
            os.environ["KAGGLE_USERNAME"] = creds["username"]
            os.environ["KAGGLE_KEY"] = creds["key"]
            print("Set KAGGLE_USERNAME and KAGGLE_KEY from .kaggle/kaggle.json.")
        except Exception as e:
            print(f"Warning: Failed to parse kaggle.json: {e}")
            
    # Set config dir to local folder as fallback
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    
    import kaggle
    
    output_dir = project_dir / "data" / "ieee" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ieee-fraud-detection dataset from Kaggle...")
    try:
        kaggle.api.competition_download_files("ieee-fraud-detection", path=str(output_dir), quiet=False)
    except Exception as e:
        print(f"Error downloading files: {e}")
        print("Please make sure you have accepted the rules of the competition at:")
        print("https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        raise e
    
    zip_path = output_dir / "ieee-fraud-detection.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path} to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete. Cleaning up zip file...")
        zip_path.unlink()
        print("Cleanup complete.")
    else:
        print("Error: Zip file not found after download.")

if __name__ == "__main__":
    download_and_extract()
