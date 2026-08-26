"""Downloads the real, openly-accessible (no auth/registration) evaluation
datasets used by notebooks/07_real_evaluation.ipynb and
notebooks/14_full_scale_evaluation.ipynb, into data/ (gitignored).

- Imagenette (fastai): a real, labeled 10-class ImageNet subset - the small-
  scale ID split used to fit the combiner/thresholds (notebooks 06-13).
- ImageNet-A / ImageNet-O (Hendrycks et al.): openly hosted, no auth needed.
- ImageNet-1k validation set + devkit: for years this project's docs stated
  the full validation set was "gated behind registration" - checked directly
  this session (HEAD request + partial-content GET confirming real tar data,
  not an HTML login page) and found to be **directly downloadable, no
  login/registration wall**. That belief is now known to be inaccurate and
  the full 50,000-image validation set is included below as the real,
  full-scale in-distribution evaluation set (DESIGN.md 11.1, 10.5).
  IMPORTANT: the ground-truth labels for this dataset are in the RAW
  ILSVRC2012 device ordering, NOT torchvision's class-index ordering - see
  scripts/imagenet1k_labels.py, which must be used to interpret them; do not
  use ILSVRC2012_validation_ground_truth.txt's integers directly as class
  indices (see that module's docstring for why).
- ImageNet-C, iNaturalist, SUN397, Places365, DTD/Textures (DESIGN.md 11.1's
  remaining shift/OOD sets): also checked and found downloadable without a
  registration wall (Zenodo record 2235448 for ImageNet-C; S3/direct HTTP for
  the rest). NOT fetched by this script - each is fetched by its own
  dedicated script (scripts/download_imagenet_c.py,
  scripts/download_ood_suite.py) since they're large enough (tens of GB, one
  multi-file Zenodo record) to warrant their own resumable, per-file
  progress/retry logic rather than fitting this script's simple one-archive-
  per-dataset loop.

Idempotent: skips any dataset whose extracted folder already exists.
"""

import os
import tarfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SOURCES = [
    {
        "name": "Imagenette (160px)",
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        "archive": "imagenette2-160.tgz",
        "extracted_dir": "imagenette2-160",
    },
    {
        "name": "ImageNet-A",
        "url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
        "archive": "imagenet-a.tar",
        "extracted_dir": "imagenet-a",
    },
    {
        "name": "ImageNet-O",
        "url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar",
        "archive": "imagenet-o.tar",
        "extracted_dir": "imagenet-o",
    },
    {
        "name": "ImageNet-1k devkit (labels)",
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        "archive": "ILSVRC2012_devkit_t12.tar.gz",
        "extracted_dir": "imagenet1k_devkit/ILSVRC2012_devkit_t12",
        "extract_into": "imagenet1k_devkit",
    },
    {
        "name": "ImageNet-1k validation images (50,000 JPEGs, 6.7GB)",
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "archive": "ILSVRC2012_img_val.tar",
        "extracted_dir": "imagenet1k_val",
        "extract_into": "imagenet1k_val",
        "min_files": 50000,
    },
]


def _content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None


def download_with_resume(url: str, dest: str, max_retries: int = 50) -> None:
    """Downloads in resumable chunks, retrying until the file actually
    reaches its expected size - not just until the socket returns EOF.

    This matters for the multi-GB ImageNet-1k/ImageNet-C/OOD archives added
    for the full-scale evaluation: a real failure mode observed this session
    is the server closing the connection early (e.g. at ~2.2GB of a 6.7GB
    file) with a clean EOF, not an exception - a naive "read until empty"
    loop silently treats that as a completed download. Verifying against
    Content-Length (fetched once via HEAD) and looping until the size
    matches is what actually makes this "resumable" rather than just
    "resumable if the one connection attempt happens to finish."
    """
    expected_size = None
    try:
        expected_size = _content_length(url)
    except Exception as e:
        print(f"  could not determine expected size via HEAD ({e}); falling back to single-attempt download")

    for attempt in range(max_retries):
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected_size is not None and existing >= expected_size:
            return
        req = urllib.request.Request(url)
        if existing:
            req.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(dest, "ab" if existing else "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            print(f"  download interrupted ({e}); retrying ({attempt + 1}/{max_retries})...")
            continue

        if expected_size is None:
            return  # can't verify completeness; trust the single attempt (old behavior)
        final_size = os.path.getsize(dest)
        if final_size >= expected_size:
            return
        print(
            f"  connection closed early at {final_size}/{expected_size} bytes "
            f"({100 * final_size / expected_size:.1f}%); retrying ({attempt + 1}/{max_retries})..."
        )

    raise RuntimeError(f"download of {url} did not complete after {max_retries} retries")


def _is_complete(source: dict) -> bool:
    extracted_path = os.path.join(DATA_DIR, source["extracted_dir"])
    if not os.path.isdir(extracted_path):
        return False
    min_files = source.get("min_files")
    if min_files is None:
        return True
    # A plain isdir check is fragile for large multi-file datasets (a folder
    # can exist from an interrupted extraction) - ImageNet-1k val specifically
    # needs exactly 50,000 JPEGs, so count them rather than trust the folder.
    n = sum(1 for _ in os.scandir(extracted_path))
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
