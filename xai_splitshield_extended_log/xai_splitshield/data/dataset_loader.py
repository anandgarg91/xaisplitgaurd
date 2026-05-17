"""
data/dataset_loader.py
──────────────────────
Unified dataset loading for CIFAR-10, GTSRB, and Chest X-ray.
Each loader returns (train_loader, test_loader, num_classes).
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


# ── Transforms ──────────────────────────────────────────────────────────────

def get_cifar10_transforms(image_size: int = 32):
    train_tf = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf


def get_gtsrb_transforms(image_size: int = 32):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669)),
    ])
    return train_tf, test_tf


def get_xray_transforms(image_size: int = 32):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    return train_tf, test_tf


# ── Dataset Builders ─────────────────────────────────────────────────────────

def _safe_download_cifar10(root: Path) -> bool:
    """
    Robustly downloads CIFAR-10 by trying multiple fallback strategies.

    Strategies (in order):
      1. Default torchvision (uses cs.toronto.edu)
      2. SSL-verification disabled (handles expired cert)
      3. HTTP URL fallback (handles HTTPS-specific cert issues)
      4. Direct Parquet download from HuggingFace (handles 503 — no extra deps)
      5. GitHub-hosted .tar.gz mirror (secondary fallback)
      6. HuggingFace `datasets` library (last resort; requires `pip install datasets`)

    Returns True if the dataset becomes available on disk in the
    torchvision-expected layout: root/cifar-10-batches-py/data_batch_*.
    """
    import ssl
    import urllib.request
    from torchvision.datasets import CIFAR10

    # Already downloaded?
    if (root / "cifar-10-batches-py" / "data_batch_1").exists():
        return True

    # ── Strategy 1: stock torchvision download ──────────────────────────────
    try:
        CIFAR10(str(root), train=True, download=True)
        return True
    except Exception as e:
        print(f"[CIFAR10] default download failed ({type(e).__name__}: {e})")

    # ── Strategy 2: disable SSL verification ────────────────────────────────
    print("[CIFAR10] retrying with SSL verification disabled...")
    _orig_ssl_context = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        CIFAR10(str(root), train=True, download=True)
        return True
    except Exception as e:
        print(f"[CIFAR10] SSL-bypass attempt failed ({type(e).__name__}: {e})")
    finally:
        ssl._create_default_https_context = _orig_ssl_context

    # ── Strategy 3: HTTP URL override ───────────────────────────────────────
    print("[CIFAR10] retrying via plain HTTP cs.toronto.edu...")
    try:
        CIFAR10.url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        ssl._create_default_https_context = ssl._create_unverified_context
        CIFAR10(str(root), train=True, download=True)
        return True
    except Exception as e:
        print(f"[CIFAR10] HTTP fallback failed ({type(e).__name__}: {e})")
    finally:
        ssl._create_default_https_context = _orig_ssl_context

    # ── Strategy 4: Direct Parquet download from HF (no `datasets` lib needed) ─
    print("[CIFAR10] trying direct Parquet download from Hugging Face...")
    try:
        if _download_cifar10_via_parquet(root):
            return True
    except Exception as e:
        print(f"[CIFAR10] Parquet download failed ({type(e).__name__}: {e})")

    # ── Strategy 5: GitHub-hosted .tar.gz mirror ────────────────────────────
    print("[CIFAR10] trying GitHub-hosted .tar.gz mirror...")
    mirror_urls = [
        "https://github.com/EN10/CIFAR/raw/master/cifar-10-python.tar.gz",
    ]
    archive_path = root / "cifar-10-python.tar.gz"
    for url in mirror_urls:
        try:
            print(f"[CIFAR10]   trying: {url}")
            ctx = ssl._create_unverified_context()
            req = urllib.request.Request(url, headers={"User-Agent": "xai-splitshield/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                data = resp.read()
            with open(archive_path, "wb") as f:
                f.write(data)
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(root)
            if (root / "cifar-10-batches-py" / "data_batch_1").exists():
                print(f"[CIFAR10]   ✓ downloaded and extracted from {url}")
                return True
        except Exception as e:
            print(f"[CIFAR10]   ✗ {type(e).__name__}: {e}")

    # ── Strategy 6: HuggingFace `datasets` library (last fallback) ──────────
    print("[CIFAR10] trying via Hugging Face datasets library...")
    try:
        if _download_cifar10_via_huggingface(root):
            return True
    except Exception as e:
        print(f"[CIFAR10] HuggingFace datasets failed ({type(e).__name__}: {e})")

    # ── All strategies failed: print manual instructions ────────────────────
    print("\n" + "=" * 70)
    print("[CIFAR10] All automatic download attempts failed.")
    print("=" * 70)
    print("Please download the dataset manually:")
    print("  Option A — Kaggle (requires account):")
    print("    https://www.kaggle.com/datasets/pankrzysiu/cifar10-python")
    print("  Option B — Hugging Face (no account needed):")
    print("    https://huggingface.co/datasets/uoft-cs/cifar10")
    print("  Option C — Original (when server is up):")
    print("    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    print()
    print(f"  Then place the .tar.gz file in:  {root.absolute()}")
    print("  And extract:  tar -xzf cifar-10-python.tar.gz")
    print(f"  The directory '{root}/cifar-10-batches-py' should appear.")
    print("=" * 70 + "\n")
    return False


def _download_cifar10_via_parquet(root: Path) -> bool:
    """
    Direct Parquet download from HuggingFace, no `datasets` library needed.

    Downloads the two verified Parquet files from the `uoft-cs/cifar10` repo
    and converts them to torchvision's pickle-batch layout:
      root/cifar-10-batches-py/data_batch_{1..5}, test_batch, batches.meta

    Requires `pyarrow` (already present via pandas in requirements.txt) and
    PIL (in requirements.txt). Returns False if those imports fail.
    """
    import ssl
    import urllib.request
    import io
    import pickle

    try:
        import pyarrow.parquet as pq
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"[CIFAR10]   missing dependency for Parquet path ({e}); skipping.")
        return False

    # Verified URLs (checked against huggingface.co/datasets/uoft-cs/cifar10 file tree)
    base = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
    urls = {
        "train": f"{base}/train-00000-of-00001.parquet",
        "test":  f"{base}/test-00000-of-00001.parquet",
    }

    out_dir = root / "cifar-10-batches-py"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = ssl._create_unverified_context()
    parquet_data = {}

    for split, url in urls.items():
        print(f"[CIFAR10]   downloading {split} parquet from HuggingFace...")
        req = urllib.request.Request(url, headers={"User-Agent": "xai-splitshield/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            parquet_data[split] = resp.read()
        size_mb = len(parquet_data[split]) / 1024 / 1024
        print(f"[CIFAR10]   ✓ {split} ({size_mb:.1f} MB)")

    def parquet_to_arrays(data: bytes):
        """Decode Parquet → (uint8 image array NHWC, int label list)."""
        table = pq.read_table(io.BytesIO(data))
        df = table.to_pandas()
        # 'img' column contains dicts {bytes: ..., path: ...} from HF datasets format
        imgs = []
        for cell in df["img"]:
            if isinstance(cell, dict):
                img_bytes = cell.get("bytes")
            else:
                img_bytes = cell
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            imgs.append(np.array(pil, dtype=np.uint8))
        return np.stack(imgs), df["label"].tolist()

    # ── Convert training set into 5 pickle batches ──────────────────────────
    train_imgs, train_labels = parquet_to_arrays(parquet_data["train"])
    assert train_imgs.shape == (50000, 32, 32, 3), \
        f"unexpected train shape {train_imgs.shape}"
    n_per_batch = 10000
    for i in range(5):
        s, e = i * n_per_batch, (i + 1) * n_per_batch
        batch_imgs = train_imgs[s:e].transpose(0, 3, 1, 2).reshape(n_per_batch, -1)
        batch = {
            b"data":        batch_imgs,
            b"labels":      train_labels[s:e],
            b"batch_label": f"training batch {i+1} of 5".encode(),
            b"filenames":   [f"img_{j:05d}.png".encode() for j in range(s, e)],
        }
        with open(out_dir / f"data_batch_{i+1}", "wb") as f:
            pickle.dump(batch, f)

    # ── Convert test set ───────────────────────────────────────────────────
    test_imgs, test_labels = parquet_to_arrays(parquet_data["test"])
    assert test_imgs.shape == (10000, 32, 32, 3), \
        f"unexpected test shape {test_imgs.shape}"
    test_imgs_flat = test_imgs.transpose(0, 3, 1, 2).reshape(10000, -1)
    test_batch = {
        b"data":        test_imgs_flat,
        b"labels":      test_labels,
        b"batch_label": b"testing batch 1 of 1",
        b"filenames":   [f"test_{j:05d}.png".encode() for j in range(10000)],
    }
    with open(out_dir / "test_batch", "wb") as f:
        pickle.dump(test_batch, f)

    # ── Metadata ───────────────────────────────────────────────────────────
    meta = {
        b"label_names": [b"airplane", b"automobile", b"bird", b"cat", b"deer",
                        b"dog", b"frog", b"horse", b"ship", b"truck"],
        b"num_cases_per_batch": 10000,
        b"num_vis": 3072,
    }
    with open(out_dir / "batches.meta", "wb") as f:
        pickle.dump(meta, f)

    print(f"[CIFAR10]   ✓ Parquet → torchvision conversion complete at {out_dir}")
    return True


def _download_cifar10_via_huggingface(root: Path) -> bool:
    """
    Last-resort fallback: use the Hugging Face `datasets` library to fetch
    CIFAR-10, then convert it to torchvision's pickle-batch format so the
    downstream torchvision CIFAR10 loader can read it.

    Requires `pip install datasets`. If unavailable, returns False without
    raising.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[CIFAR10]   `datasets` library not installed; skipping HF fallback.")
        print("[CIFAR10]   Install with: pip install datasets")
        return False

    import pickle
    import numpy as np
    from PIL import Image

    # The uoft-cs/cifar10 repo serves the dataset in standard format
    print("[CIFAR10]   loading via datasets.load_dataset('uoft-cs/cifar10')...")
    ds = load_dataset("uoft-cs/cifar10")

    out_dir = root / "cifar-10-batches-py"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Write training batches ─────────────────────────────────────────────
    train_imgs   = np.array([np.array(im).astype(np.uint8)
                             for im in ds["train"]["img"]])
    train_labels = list(ds["train"]["label"])
    n_per_batch  = 10000
    for i in range(5):
        start, end = i * n_per_batch, (i + 1) * n_per_batch
        batch_imgs = train_imgs[start:end].transpose(0, 3, 1, 2).reshape(n_per_batch, -1)
        batch = {
            b"data":         batch_imgs,
            b"labels":       train_labels[start:end],
            b"batch_label":  f"training batch {i+1} of 5".encode(),
            b"filenames":    [f"img_{j:05d}.png".encode() for j in range(start, end)],
        }
        with open(out_dir / f"data_batch_{i+1}", "wb") as f:
            pickle.dump(batch, f)

    # ── Write test batch ───────────────────────────────────────────────────
    test_imgs   = np.array([np.array(im).astype(np.uint8)
                            for im in ds["test"]["img"]])
    test_labels = list(ds["test"]["label"])
    test_imgs_flat = test_imgs.transpose(0, 3, 1, 2).reshape(len(test_imgs), -1)
    test_batch = {
        b"data":        test_imgs_flat,
        b"labels":      test_labels,
        b"batch_label": b"testing batch 1 of 1",
        b"filenames":   [f"test_{j:05d}.png".encode() for j in range(len(test_imgs))],
    }
    with open(out_dir / "test_batch", "wb") as f:
        pickle.dump(test_batch, f)

    # ── Metadata file ──────────────────────────────────────────────────────
    meta = {
        b"label_names": [b"airplane", b"automobile", b"bird", b"cat", b"deer",
                        b"dog", b"frog", b"horse", b"ship", b"truck"],
        b"num_cases_per_batch": 10000,
        b"num_vis": 3072,
    }
    with open(out_dir / "batches.meta", "wb") as f:
        pickle.dump(meta, f)

    print(f"[CIFAR10]   ✓ converted HF dataset to torchvision layout at {out_dir}")
    return True


def load_cifar10(
    root: str,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_tf, test_tf = get_cifar10_transforms(image_size)

    if not _safe_download_cifar10(root):
        raise RuntimeError(
            "CIFAR-10 dataset is not available on disk and auto-download "
            "failed. See the manual instructions above."
        )

    # download=False here because _safe_download_cifar10 already succeeded
    train_ds = datasets.CIFAR10(str(root), train=True,  download=False, transform=train_tf)
    test_ds  = datasets.CIFAR10(str(root), train=False, download=False, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10


def _safe_download_gtsrb(root: Path) -> bool:
    """
    Robust GTSRB downloader with SSL fallback. The German archive's HTTPS
    cert occasionally has issues; this disables verification on retry.
    """
    import ssl
    from torchvision.datasets import GTSRB

    # Already downloaded?
    if (root / "gtsrb" / "GTSRB").exists():
        return True

    try:
        GTSRB(str(root), split="train", download=True)
        GTSRB(str(root), split="test",  download=True)
        return True
    except Exception as e:
        print(f"[GTSRB] default download failed ({type(e).__name__}: {e})")
        print("[GTSRB] retrying with SSL verification disabled...")

    _orig_ctx = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        GTSRB(str(root), split="train", download=True)
        GTSRB(str(root), split="test",  download=True)
        return True
    except Exception as e:
        print(f"[GTSRB] SSL-bypass attempt failed ({type(e).__name__}: {e})")
    finally:
        ssl._create_default_https_context = _orig_ctx

    print("\n" + "=" * 70)
    print("[GTSRB] Automatic download failed. Manual instructions:")
    print("  Download the GTSRB archive from:")
    print("    https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/")
    print(f"  Extract into:  {root.absolute()}/gtsrb/")
    print("=" * 70 + "\n")
    return False


def load_gtsrb(
    root: str,
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_tf, test_tf = get_gtsrb_transforms(image_size)

    if not _safe_download_gtsrb(root):
        raise RuntimeError(
            "GTSRB is not available on disk and auto-download failed. "
            "See manual instructions above."
        )

    train_ds = datasets.GTSRB(str(root), split="train", download=False, transform=train_tf)
    test_ds  = datasets.GTSRB(str(root), split="test",  download=False, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 43


def load_xray(
    root: str,
    image_size: int = 32,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Chest X-Ray dataset (Kermany et al., Cell 2018).
    Expected directory layout:
      root/
        train/  NORMAL/  *.jpeg
                PNEUMONIA/  *.jpeg
        test/   NORMAL/  *.jpeg
                PNEUMONIA/  *.jpeg
    Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    """
    root = Path(root) / "chest_xray"
    train_tf, test_tf = get_xray_transforms(image_size)

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    test_ds  = datasets.ImageFolder(root / "test",  transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 2


# ── Dispatcher ───────────────────────────────────────────────────────────────

def get_dataset(
    name: str,
    root: str = "./data/raw",
    image_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Returns (train_loader, test_loader, num_classes) for the requested dataset.
    """
    loaders = {
        "cifar10": load_cifar10,
        "gtsrb":   load_gtsrb,
        "xray":    load_xray,
    }
    name = name.lower()
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders)}")
    return loaders[name](root, image_size, batch_size, num_workers)
