# Puzzle-Joiner

Python/Qt desktop app for assembling puzzle pieces from scanned PDFs or images. Split into a `puzzle_joiner/` package, launched via `./puzzle-joiner` (uv script).

## Tech Stack

- **Python 3.11+**, packaged with `uv` (inline script dependencies in `puzzle-joiner`)
- **PySide6** (Qt6) for UI
- **OpenCV** for image processing and feature matching
- **NumPy**, **tifffile** for array manipulation and TIFF export
- **Poppler** (`pdfseparate`, `pdftocairo`, `pdfinfo`) and **ImageMagick** (`convert`) for PDF handling

## File Structure

```
puzzle-joiner              # Entry script (uv shebang, deps, thin launcher)
Makefile                   # make check: syntax, unit tests, boot test
puzzle_joiner/
  __init__.py              # Empty
  __main__.py              # main() entry point (also supports python -m puzzle_joiner)
  priority.py              # LOW_PRIO prefix, _set_low_priority() for nice/ionice/chrt
  model.py                 # PuzzlePiece data model
  preprocessing.py         # PDF split, border crop, legend removal, auto-crop computation
  cache.py                 # Page cache (~/.cache/puzzle-joiner/{md5}/)
  matching.py              # ORB features, pairwise matching, auto-detect, snap
  export.py                # Layered + flattened TIFF export
  undo.py                  # QUndoCommand subclasses (TransformCommand, SnapCommand, LockCommand)
  worker.py                # WorkerSignals, Worker (QRunnable pattern)
  widgets.py               # All Qt widgets (handles, piece items, scene, view, thumbnails, dialogs)
  main_window.py           # MainWindow (toolbar, import flow, action handlers)
tests/
  test_preprocessing.py    # parse_page_range tests
  test_matching.py         # decompose_affine tests
  test_model.py            # PuzzlePiece math (crop, affine, bounding box)
  test_cache.py            # cache path generation
  test_priority.py         # LOW_PRIO prefix
  test_undo.py             # undo/redo commands (transform, snap, lock)
  test_boot.py             # smoke test: MainWindow constructs (offscreen Qt)
```

## Module Responsibilities

### `model.py`
`PuzzlePiece` — holds full-res BGRA image, display pixmap (max 2048px), thumbnail (256px), and transform state (`x, y, rotation_deg, scale, crop_rect`). Key methods: `get_affine_matrix()`, `get_bounding_box()`, `get_cropped_image()`.

### `preprocessing.py`
Pipeline: `pdfseparate` → ImageMagick convert at 300 DPI → border line detection via Hough transform (`autocrop_border_lines`) → legend/margin removal (`remove_legend`) → auto-crop computation (`compute_auto_crop_rect`). Also contains `_convert_and_process_page` worker for ProcessPoolExecutor. Imports `LOW_PRIO` from `priority`.

### `cache.py`
Persistent cache at `~/.cache/puzzle-joiner/{md5}/`. Stores uncropped page PNGs (`page_XXXX.png`) and auto-crop rects (`page_XXXX_autocrop.json`). `_load_cached_page` reads or recomputes crop rect on load. Imports `compute_auto_crop_rect` from `preprocessing`.

### `matching.py`
- **ORB extraction** (`compute_orb_features`): per-channel (B,G,R), optional gray masking, WTA_K=4.
- **Pairwise matching** (`estimate_pairwise_transform`): BFMatcher cross-check → `estimateAffinePartial2D`. Pixel-level validation.
- **Auto-detect** (`auto_detect_placements`): greedy BFS placement, >75% pixel match threshold.
- **Snap** (`snap_piece_to_neighbors`): 50k features, no gray masking, ROI-based overlap matching, ≥50% pixel threshold.

### `export.py`
- **Layered TIFF**: each piece as a separate BigTIFF layer (deflate, 1024x1024 tiles).
- **Flattened TIFF**: alpha-composite all pieces onto single canvas.

### `widgets.py`
- `HandleItem`, `TransformHandles`: GIMP-style resize/rotate handles.
- `PuzzlePieceItem`: syncs between model and scene, red border when selected.
- `PuzzleScene`, `PuzzleView`: virtual canvas with zoom/pan.
- `ThumbnailPanel`: horizontal strip with lock/matched indicators.
- `PdfImportDialog`: page selection grid with async thumbnail generation.
- `CutToolDialog`: crop region editor showing full uncropped image.

### `main_window.py`
Toolbar workflow: Open PDF/Images → Auto-Detect → Snap → Cut → Lock → Export. Manages piece list, threading, progress dialogs.

### `priority.py`
Builds `LOW_PRIO` prefix (`chrt --idle 0 nice -n19 ionice --class idle`) for subprocesses. `_set_low_priority()` for ProcessPoolExecutor worker initializer.

## Key Data Flow

```
PDF/Images → convert & clean (parallel, low-prio) → cache → PuzzlePiece list
  → Auto-Detect (ORB matching, BFS placement) → scene positions
  → Manual adjust / Snap / Cut → Export TIFF
```

## Running

```
uv run puzzle-joiner
# or
python -m puzzle_joiner
```

## Checking & Testing

Run `make check` to verify syntax, lint, unit tests, and boot test. Do this instead of manually checking syntax.

```
make check          # all checks: syntax + lint + test + boot
make syntax         # Python AST parse of all .py files
make lint           # pyflakes undefined-name check
make test           # unit tests (pytest, no display needed)
make boot           # smoke test: MainWindow constructs (offscreen Qt)
```

Tests live in `tests/` and cover pure functions (`parse_page_range`, `decompose_affine`, `PuzzlePiece` model math, cache paths, priority). The boot test verifies the full import chain and MainWindow construction.
