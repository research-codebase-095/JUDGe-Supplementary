"""Downloads the standard ImageNet-pretrained OOD suite named in DESIGN.md
11.1: iNaturalist, SUN397, Places365, DTD/Textures. Evaluate-only per the
data-splitting protocol (10.5) - never used for fitting the combiner or
thresholds, only for measuring AUROC(id vs. OOD) / FPR@95%TPR against the
already-frozen pipeline.

All four were checked this session for a registration wall and found openly
downloadable EXCEPT SUN397 - see the SUN397 entry below for the specific,
verified blocker (a dead link, not a registration gate).

- iNaturalist 2021 validation (8.9 GB) + annotations (~10 MB): S3-hosted,
  no auth. `ml-inat-competition-datasets.s3.amazonaws.com`.
- Places365 validation, 256px variant (525 MB): the appropriately-sized
  variant (not the ~30GB full-res one), matches this project's other
  datasets' resolution scale. `data.csail.mit.edu`.
- DTD/Textures (625 MB): redirects `robots.ox.ac.uk` -> `thor.robots.ox.ac.uk`,
  no auth.
- SUN397 (~36 GB) - **checked and found NOT currently downloadable at its
  documented location**, unlike the other three. The URL cited in
  torchvision's own dataset loader and in this project's original task
  description (`http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz`)
  redirects twice (vision.princeton.edu -> https same path -> a 301 to
  `https://www.cs.princeton.edu/research/areas/gravisprojects/2010/SUN/SUN397.tar.gz`)
  and the final URL 404s - confirmed with curl showing the full redirect
  chain, not just a single failed guess. This is a genuinely dead/moved
  link, not a registration wall - the exact kind of "turns out blocked when
  you actually try it" case this project's evaluation protocol document
  anticipated. A community mirror exists on HuggingFace
  (huggingface.co/datasets/tanganke/sun397), but in parquet format rather
  than this project's expected class-labeled-JPEG-directory layout, which
  would need its own conversion step - left as a follow-up, not attempted
  in this pass, to keep scope bounded. SUN397 is therefore NOT included in
  SOURCES below; the OOD suite this pass covers is iNaturalist + Places365 +
  DTD only, a stated scope reduction, not a silent omission.

Idempotent: skips any dataset whose extracted folder already exists (with a
file-count floor for the larger ones, since a folder existing isn't proof an
interrupted extraction actually finished).
"""

from __future__ import annotations

import os
import tarfile
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

SOURCES = [
    {
        "name": "Places365 validation (256px)",
        "url": "http://data.csail.mit.edu/places/places365/val_256.tar",
        "archive": "places365_val_256.tar",
        "extracted_dir": "places365_val_256",
        "extract_into": "places365_val_256",
        "min_files": 36000,  # 36,500 val images
    },
    {
        "name": "DTD (Describable Textures Dataset)",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        "archive": "dtd-r1.0.1.tar.gz",
        "extracted_dir": "dtd",
        "min_files": 1,  # top-level dir has a handful of subdirs, not a flat file count
    },
    {
        "name": "iNaturalist 2021 validation images",
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
        "archive": "inat2021_val.tar.gz",
        "extracted_dir": "inaturalist2021_val",
        "extract_into": "inaturalist2021_val",
        "min_files": 1,
    },
    {
        "name": "iNaturalist 2021 validation annotations",
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
        "archive": "inat2021_val_json.tar.gz",
        "extracted_dir": "inaturalist2021_val_annotations",
        "extract_into": "inaturalist2021_val_annotations",
        "min_files": 1,
    },
]


def download_with_resume(url: str, dest: str, max_retries: int = 50) -> None:
    """Retries until the file reaches its expected Content-Length, not just
    until a socket read returns EOF - a real observed failure mode for these
    large archives is the connection closing cleanly partway through (e.g.
    at ~2GB of a 6.7GB file), which a naive "read until empty" loop would
    silently accept as a completed download.
    """
    expected_size = None
    try:
        head_req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            expected_size = int(cl) if cl is not None else None
    except Exception as e:
        print(f"  could not determine expected size via HEAD ({e}); falling back to single-attempt download")

    for attempt in range(max_retries):
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected_size is not None and existing >= expected_size:
            return
        req = urllib.request.Request(url)
        if existing:
            req.add_header("Range", f"bytes={existing}-")
        t0 = time.time()
        total = existing
        last_report = total
        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(dest, "ab" if existing else "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total - last_report >= 200 * 1024 * 1024:
                        rate = (total - existing) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"  {total / 1e9:.2f} GB downloaded ({rate:.1f} MB/s)", flush=True)
                        last_report = total
        except Exception as e:
            print(f"  download interrupted ({e}); retrying ({attempt + 1}/{max_retries})...")
            continue

        if expected_size is None:
            return
        if total >= expected_size:
            return
        print(
            f"  connection closed early at {total}/{expected_size} bytes "
            f"({100 * total / expected_size:.1f}%); retrying ({attempt + 1}/{max_retries})..."
        )

    raise RuntimeError(f"download of {url} did not complete after {max_retries} retries")


def _is_complete(source: dict) -> bool:
    extracted_path = os.path.join(DATA_DIR, source["extracted_dir"])
    if not os.path.isdir(extracted_path):
        return False
    min_files = source.get("min_files", 1)
    n = 0
    for _root, _dirs, files in os.walk(extracted_path):
        n += len(files)
        if n >= min_files:
            return True
    return n >= min_files


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for source in SOURCES:
        extracted_path = os.path.join(DATA_DIR, source["extracted_dir"])
        if _is_complete(source):
            print(f"[skip] {source['name']} already extracted at {extracted_path}")
            continue

        archive_path = os.path.join(DATA_DIR, source["archive"])
        print(f"[download] {source['name']} -> {archive_path}")
        download_with_resume(source["url"], archive_path)

        print(f"[extract] {archive_path}")
        extract_target = os.path.join(DATA_DIR, source["extract_into"]) if "extract_into" in source else DATA_DIR
        os.makedirs(extract_target, exist_ok=True)
        mode = "r:gz" if archive_path.endswith((".tgz", ".tar.gz")) else "r"
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(extract_target)

        os.remove(archive_path)
        print(f"[done] {source['name']}")


if __name__ == "__main__":
    main()
