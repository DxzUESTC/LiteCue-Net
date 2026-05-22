"""Download insightface buffalo_l into models/buffalo_l/ (fallback only).

The API normally uses models/buffalo_l/ bundled in this repo (Git LFS).
Run this script only when LFS was not pulled or files are missing:

    git lfs install && git lfs pull   # preferred
    python scripts/download_models.py # fallback, needs network
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same URL insightface itself uses (insightface/utils/storage.py)
BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
MODEL_NAME = "buffalo_l"
EXPECTED_FILES = [
    "det_10g.onnx",
    "w600k_r50.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a progress bar."""
    try:
        import requests
    except ImportError:
        print("error: 'requests' is required.  Install it first:")
        print("    pip install requests")
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        print("error: 'tqdm' is required.  Install it first:")
        print("    pip install tqdm")
        sys.exit(1)

    print(f"Downloading {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract *zip_path* into *target_dir*, flattening a possible top-level
    directory whose name matches ``MODEL_NAME``.

    Some distributions of buffalo_l.zip contain a ``buffalo_l/`` prefix inside
    the archive; this method strips it so that files land directly in
    *target_dir*.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Determine the common prefix (if any)
        prefix = os.path.commonpath(names)
        if prefix and prefix != "." and prefix != MODEL_NAME:
            # Unexpected prefix — extract as-is
            zf.extractall(target_dir)
            return

        # Strip the single top-level directory
        prefix_len = len(prefix) + 1 if prefix else 0  # +1 for trailing /
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel_path = member.filename[prefix_len:]
            if not rel_path:
                continue
            member.filename = rel_path
            zf.extract(member, target_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def download_models(models_dir: Path, force: bool = False) -> None:
    """Download and extract ``buffalo_l`` into ``models_dir / buffalo_l``."""
    target = models_dir / MODEL_NAME

    # ---- Already exists? ----
    if target.is_dir() and not force:
        missing = [f for f in EXPECTED_FILES if not (target / f).is_file()]
        if not missing:
            print(f"[✓] {MODEL_NAME} already present at {target}")
            print(f"    ({len(EXPECTED_FILES)}/{len(EXPECTED_FILES)} ONNX files found)")
            return
        print(f"Missing files: {missing}; re-downloading ...")

    # ---- Download ----
    target.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / f"{MODEL_NAME}.zip"
    try:
        _download(BUFFALO_L_URL, zip_path)
        _extract_zip(zip_path, target)

        # ---- Verify ----
        present = [f for f in EXPECTED_FILES if (target / f).is_file()]
        print(
            f"[✓] {len(present)}/{len(EXPECTED_FILES)} files downloaded to {target}"
        )
        if len(present) != len(EXPECTED_FILES):
            print(f"    Expected: {EXPECTED_FILES}")
            print(f"    Found:    {present}")
            print("    Some files are missing — the model may not work correctly.")
    finally:
        if zip_path.exists():
            zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pretrained models to the project-local models/ directory"
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Path to the models/ directory (default: <project-root>/models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model already exists",
    )
    args = parser.parse_args()

    if args.models_dir is None:
        base = Path(__file__).resolve().parent.parent
        models_dir = base / "models"
    else:
        models_dir = Path(args.models_dir).resolve()

    download_models(models_dir, force=args.force)


if __name__ == "__main__":
    main()
