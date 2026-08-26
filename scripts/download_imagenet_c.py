"""Downloads ImageNet-C (Hendrycks & Dietterich, 2019) - the corruption /
distribution-shift test set named in DESIGN.md 11.1, evaluate-only per the
data-splitting protocol (10.5): never used for fitting the combiner or
thresholds, only for measuring how the already-frozen pipeline generalizes.

Source: Zenodo record 2235448 (https://zenodo.org/records/2235448) - checked
this session via the record's own API (https://zenodo.org/api/records/2235448)
rather than guessing filenames (a guessed "noise.tar" URL at a different path
returned a tiny non-file response when tried). The record splits ImageNet-C
into five tars by corruption category, each containing severities 1-5 for
every corruption in that category:

    blur.tar     ~7.1 GB  (defocus, glass, motion, zoom blur)
    weather.tar ~12.8 GB  (snow, frost, fog, brightness)
    digital.tar  ~7.8 GB  (contrast, elastic, pixelate, jpeg)
    noise.tar   ~22.6 GB  (gaussian, shot, impulse, speckle noise)
    extra.tar   ~15.8 GB  (extra corruptions not in the original 15; optional -
                           not part of the standard 15-corruption benchmark,
                           included here for completeness but the standard
                           mean-corruption-error protocol only needs the
                           first four)

Total ~66 GB across the four standard-benchmark tars (~82GB with extra.tar) -
by a wide margin the largest single dataset this project has ever attempted
to acquire, and, at this environment's measured ~5 MB/s download rate, a
multi-hour download even for one tar. This script downloads and extracts one
tar at a time, resumable per-file, so a partial run leaves genuinely completed
categories in place rather than an all-or-nothing state.

Usage: python scripts/download_imagenet_c.py [category ...]
    (no args -> attempts blur, weather, digital, noise, in that size order,
    smallest first, so a time-limited run banks as much real coverage as
    possible; pass e.g. "blur digital" to fetch only specific categories;
    "extra" is never included by default - pass it explicitly if wanted.)
"""

from __future__ import annotations

import os
import sys
import tarfile
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
IMAGENET_C_DIR = os.path.join(DATA_DIR, "imagenet-c")

ZENODO_RECORD_API = "https://zenodo.org/api/records/2235448"

# Smallest-first default order (see module docstring for the reasoning).
DEFAULT_CATEGORIES = ["blur", "digital", "weather", "noise"]
ALL_CATEGORIES = ["blur", "digital", "weather", "noise", "extra"]


def get_file_urls() -> dict[str, tuple[str, int]]:
    """Fetches the record's real file list + sizes from the Zenodo API rather
    than guessing URLs (see module docstring)."""
    import json

    req = urllib.request.Request(ZENODO_RECORD_API)
    with urllib.request.urlopen(req, timeout=30) as r:
        record = json.load(r)
    out = {}
    for f in record["files"]:
        key = f["key"]  # e.g. "blur.tar"
        category = key.replace(".tar", "")
        out[category] = (f["links"]["self"], int(f["size"]))
    return out


def download_with_resume(url: str, dest: str, expected_size: int | None = None, max_retries: int = 50) -> None:
    """Retries until `dest` actually reaches `expected_size` (known here from
    the Zenodo API's own file listing, so no separate HEAD is needed) rather
    than trusting a single connection's EOF - a real observed failure mode
    for these very large tars is the connection closing cleanly partway
    through, which a naive "read until empty" loop would silently accept as
    a completed download.
    """
    for attempt in range(max_retries):
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected_size is not None and existing >= expected_size:
            if attempt == 0:
                print(f"  already fully downloaded ({existing} bytes)")
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


def main(categories: list[str]) -> None:
    os.makedirs(IMAGENET_C_DIR, exist_ok=True)
    print("fetching real file list from Zenodo API...")
    file_urls = get_file_urls()

    for category in categories:
        if category not in file_urls:
            print(f"[skip] unknown category {category!r}; available: {sorted(file_urls)}")
            continue
        marker = os.path.join(IMAGENET_C_DIR, f".{category}_done")
        if os.path.exists(marker):
            print(f"[skip] {category} already extracted (marker present)")
            continue

        url, size = file_urls[category]
        archive_path = os.path.join(DATA_DIR, f"imagenet-c-{category}.tar")
        print(f"[download] {category} ({size / 1e9:.2f} GB) -> {archive_path}")
        download_with_resume(url, archive_path, expected_size=size)

        print(f"[extract] {archive_path} -> {IMAGENET_C_DIR}")
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(IMAGENET_C_DIR)

        os.remove(archive_path)
        with open(marker, "w") as f:
            f.write("done\n")
        print(f"[done] {category}")


if __name__ == "__main__":
    cats = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CATEGORIES
    main(cats)
