
import os
import requests
import zipfile
import io
from pathlib import Path

def download_df_model(target_dir: str = "models-cache", model_name: str = "DeepFilterNet3"):
    """
    Downloads and extracts the DeepFilterNet model to the specified directory.
    """
    # Create target directory
    target_path = Path(target_dir) / model_name
    if target_path.exists():
        print(f"Model already exists at {target_path}")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    url = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{model_name}.zip"
    print(f"Downloading {model_name} from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        print("Download complete. Extracting...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_dir)
            
        print(f"Model extracted to {target_dir}")
        
    except Exception as e:
        print(f"Failed to download/extract model: {e}")
        # Clean up partial download if needed (though here we do memory extract)
        raise

if __name__ == "__main__":
    # Ensure run from server root or adjust path
    base_dir = Path(__file__).parent.parent
    cache_dir = base_dir / "models-cache"
    download_df_model(str(cache_dir))
