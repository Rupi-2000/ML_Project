import os
import urllib.request
import zipfile
from pathlib import Path

# =========================
# Configuration
# =========================

DATA_DIR = Path("data")
EXTRA_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_extra.zip"
)

CORE_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_core.zip"
)

EXTRA_DIR = DATA_DIR / "doclaynet_extra"
CORE_DIR = DATA_DIR / "doclaynet_core"



# =========================
# Utility functions
# =========================

def download(url: str, target: Path):
    if target.exists():
        print(f"[OK] {target.name} already exists – skipping download")
        return

    print(f"[DOWNLOAD] {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    print(f"[DONE] Downloaded to {target}")


def extract(zip_path: Path, target_dir: Path):
    if target_dir.exists():
        print(f"[OK] {target_dir} already exists – skipping extraction")
        return

    print(f"[EXTRACT] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"[DONE] Extracted to {target_dir}")


# =========================
# Main logic
# =========================

def prepare_core():
    print("\n=== Preparing DocLayNet CORE (YOLO / Layout) ===")
    zip_path = DATA_DIR / "DocLayNet_core.zip"
    download(CORE_URL, zip_path)
    extract(zip_path, CORE_DIR)


def prepare_extra():
    print("\n=== Preparing DocLayNet EXTRA (Text / TF-IDF) ===")
    zip_path = DATA_DIR / "DocLayNet_extra.zip"
    download(EXTRA_URL, zip_path)
    extract(zip_path, EXTRA_DIR)


def main():
    prepare_extra()
    prepare_core()
    

    print("\n=== Dataset preparation complete ===")
    print("Extra:", EXTRA_DIR.resolve())
    print("Core :", CORE_DIR.resolve())


if __name__ == "__main__":
    main()
