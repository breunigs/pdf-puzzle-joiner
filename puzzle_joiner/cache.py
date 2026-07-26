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


def _layout_path(cache_dir: str) -> str:
    """Path for the layout JSON file in a cache directory."""
    return os.path.join(cache_dir, "layout.json")


def save_layout(cache_dir: str, pieces, snap_pairs=None) -> None:
    """Save piece layout state and snap pairs to cache dir."""
    if not cache_dir:
        return
    entries = []
    for piece in pieces:
        entries.append({
            "source": os.path.basename(piece.source_path),
            "x": piece.x,
            "y": piece.y,
            "rotation_deg": piece.rotation_deg,
            "scale": piece.scale,
            "is_matched": piece.is_matched,
            "is_locked": piece.is_locked,
            "crop_rect": list(piece.crop_rect) if piece.crop_rect is not None else None,
        })
    data = {"pieces": entries}
    if snap_pairs:
        data["snap_pairs"] = [
            [os.path.basename(a.source_path), os.path.basename(b.source_path)]
            for a, b in snap_pairs
        ]
    path = _layout_path(cache_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_layout(cache_dir: str, pieces):
    """Restore piece layout state from cache dir.

    Returns (applied: bool, snap_pairs: list of [basename_a, basename_b]).
    """
    if not cache_dir:
        return False, []
    path = _layout_path(cache_dir)
    if not os.path.exists(path):
        return False, []
    with open(path, "r") as f:
        raw = json.load(f)
    # Handle both old format (flat list) and new format (dict with pieces/snap_pairs)
    if isinstance(raw, list):
        entries = raw
        raw_snap_pairs = []
    else:
        entries = raw.get("pieces", [])
        raw_snap_pairs = raw.get("snap_pairs", [])
    by_source = {e["source"]: e for e in entries}
    applied = False
    for piece in pieces:
        key = os.path.basename(piece.source_path)
        e = by_source.get(key)
        if e is None:
            continue
        piece.x = e["x"]
        piece.y = e["y"]
        piece.rotation_deg = e["rotation_deg"]
        piece.scale = e["scale"]
        piece.is_matched = e.get("is_matched", False)
        piece.is_locked = e.get("is_locked", False)
        saved_crop = e.get("crop_rect")
        if saved_crop is not None:
            piece.crop_rect = tuple(saved_crop)
        applied = True
    return applied, raw_snap_pairs


def _images_cache_dir(image_paths: list) -> str:
    """Return a cache directory for a set of image paths, creating it if needed."""
    md5 = hashlib.md5()
    for p in sorted(image_paths):
        md5.update(p.encode())
    d = os.path.join(CACHE_BASE, "images_" + md5.hexdigest())
    os.makedirs(d, exist_ok=True)
    return d


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
