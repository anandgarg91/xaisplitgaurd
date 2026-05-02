"""
scripts/download_datasets.py
─────────────────────────────
Standalone helper to pre-download CIFAR-10 and GTSRB before training.

Use this if your training environment has network restrictions, or if the
torchvision auto-downloader keeps failing — this script applies SSL-bypass
strategies and clear error reporting.

Usage:
    python scripts/download_datasets.py                      # download all
    python scripts/download_datasets.py --datasets cifar10   # one only
    python scripts/download_datasets.py --root ./data/raw    # custom path
"""

from __future__ import annotations
import argparse
import sys
import urllib.request
import ssl
import tarfile
from pathlib import Path

# Ensure we can import the project's loaders
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASETS = ["cifar10", "gtsrb", "xray"]


# ── Manual file-level downloader (last-resort fallback) ─────────────────────

def manual_download_cifar10(root: Path) -> bool:
    """Direct urllib download with SSL disabled — bypasses torchvision entirely."""
    archive = root / "cifar-10-python.tar.gz"
    target  = root / "cifar-10-batches-py"

    if target.exists() and (target / "data_batch_1").exists():
        print(f"[CIFAR10] Already extracted at {target}")
        return True

    if not archive.exists():
        urls = [
            # GitHub-hosted mirror (reliable; CDN-backed)
            "https://github.com/EN10/CIFAR/raw/master/cifar-10-python.tar.gz",
            # Original (often returns 503)
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        ]
        ctx = ssl._create_unverified_context()
        for url in urls:
            try:
                print(f"[CIFAR10] Downloading {url} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "xai-splitshield/1.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                    with open(archive, "wb") as f:
                        f.write(resp.read())
                print(f"[CIFAR10] ✓ Saved → {archive}")
                break
            except Exception as e:
                print(f"[CIFAR10] ✗ {type(e).__name__}: {e}")
        else:
            print("[CIFAR10] All .tar.gz mirrors failed.")
            print("[CIFAR10] Will try the Parquet path via the dataset_loader fallback chain...")
            return False

    print(f"[CIFAR10] Extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    print(f"[CIFAR10] Done. Available at {target}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="./data/raw",
                        help="Root directory for downloaded data")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS,
                        help="Which datasets to download")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    success = True

    if "cifar10" in args.datasets:
        print("\n" + "=" * 70)
        print(" CIFAR-10")
        print("=" * 70)
        # First try torchvision-based loader (handles all SSL fallbacks)
        try:
            from data.dataset_loader import _safe_download_cifar10
            if not _safe_download_cifar10(root):
                # Last resort: pure urllib
                if not manual_download_cifar10(root):
                    success = False
        except Exception as e:
            print(f"[CIFAR10] loader-based download failed: {e}")
            print("[CIFAR10] falling back to direct urllib download...")
            if not manual_download_cifar10(root):
                success = False

    if "gtsrb" in args.datasets:
        print("\n" + "=" * 70)
        print(" GTSRB")
        print("=" * 70)
        try:
            from data.dataset_loader import _safe_download_gtsrb
            if not _safe_download_gtsrb(root):
                success = False
        except Exception as e:
            print(f"[GTSRB] download failed: {e}")
            success = False

    if "xray" in args.datasets:
        print("\n" + "=" * 70)
        print(" Chest X-Ray (Kermany et al., Cell 2018)")
        print("=" * 70)
        xray_path = root / "chest_xray"
        if xray_path.exists() and (xray_path / "train").exists():
            print(f"[XRAY] Already present at {xray_path}")
        else:
            print("[XRAY] This dataset is NOT in torchvision and must be")
            print("       downloaded manually from Kaggle:")
            print("         https://www.kaggle.com/datasets/paultimothymooney/")
            print("         chest-xray-pneumonia")
            print(f"       Then extract into:  {xray_path.absolute()}")
            print("       Expected layout after extract:")
            print("         chest_xray/train/{NORMAL,PNEUMONIA}/*.jpeg")
            print("         chest_xray/test/{NORMAL,PNEUMONIA}/*.jpeg")
            success = False

    print()
    if success:
        print("✓ All requested datasets are ready.")
    else:
        print("✗ Some datasets are not yet available. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
