import os
import hashlib
import json
import shutil
import time

import cv2

from .preprocessing import compute_auto_crop_rect


CACHE_BASE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "puzzle-joiner")
_CACHE_MAX_AGE_SECONDS = 30 * 24 * 3600  # 1 month


def cleanup_old_cache_entries():
    """Remove cache entries (subdirectories of CACHE_BASE) older than 1 month."""
    if not os.path.isdir(CACHE_BASE):
        return
    now = time.time()
    for entry in os.listdir(CACHE_BASE):
        entry_path = os.path.join(CACHE_BASE, entry)
        if not os.path.isdir(entry_path):
            continue
        if now - os.path.getmtime(entry_path) > _CACHE_MAX_AGE_SECONDS:
            shutil.rmtree(entry_path, ignore_errors=True)


def _pdf_cache_key(pdf_path: str) -> str:
    """Deterministic cache key from md5sum of file content."""
    md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _cache_dir_for_pdf(pdf_path: str) -> str:
    """Return cache directory for a given PDF, creating it if needed."""
    key = _pdf_cache_key(pdf_path)
    d = os.path.join(CACHE_BASE, key)
    os.makedirs(d, exist_ok=True)
    return d


def _cached_page_path(cache_dir: str, page_num: int) -> str:
    """Path for a cached processed page (1-indexed)."""
    return os.path.join(cache_dir, f"page_{page_num:04d}.png")


def _cached_autocrop_path(cache_dir: str, page_num: int) -> str:
    """Path for a cached auto-crop rect JSON (1-indexed)."""
    return os.path.join(cache_dir, f"page_{page_num:04d}_autocrop.json")


def _load_cached_page(args):
    """Worker: load cached page PNG and read or compute auto-crop rect.

    Args:
        args: (page_num, cache_path, idx)

    Returns:
        (cache_path, bgra_array, crop_rect_or_None, idx) or None
    """
    page_num, cp, idx = args
    bgra = cv2.imread(cp, cv2.IMREAD_UNCHANGED)
    if bgra is None:
        return None
    cache_dir = os.path.dirname(cp)
    ac_path = _cached_autocrop_path(cache_dir, page_num)
    crop_rect = None
    if os.path.exists(ac_path):
        with open(ac_path, "r") as f:
            data = json.load(f)
        if data is not None:
            crop_rect = tuple(data)
    else:
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        crop_rect = compute_auto_crop_rect(bgr)
        with open(ac_path, "w") as f:
            json.dump(crop_rect, f)
    return (cp, bgra, crop_rect, idx)
