import os
import zipfile
import requests
from tqdm import tqdm

# === CONFIGURATION ===
dropbox_url = "https://www.dropbox.com/scl/fi/yhfvy67bdn4dbwgqloe3i/videos.zip?rlkey=wjg3u7idhztvg7t89wckt1wjs&st=q34lstzx&raw=1"
output_zip_path = "data/data_bundle.zip"
extract_dir = "data/"



# === DOWNLOAD ===
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

# === UNZIP ===
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        # ensure it worked
        if not os.path.exists(extract_to):
            raise FileNotFoundError(f"Extraction failed: {extract_to} does not exist.")
        else:
            # list files in the directory to confirm extraction
            extracted_files = os.listdir(extract_to)
            print(f"Extracted files: {extracted_files}")
            print(f"✅ Unzipped to: {extract_to}")
            # delete the zip file after extraction
            os.remove(zip_path)

# === EXECUTION ===
if __name__ == "__main__":
    print("📥 Downloading zip file ...")
    download_file(dropbox_url, output_zip_path)

    print("📦 Unzipping zip file ...")
    unzip_file(output_zip_path, extract_dir)

