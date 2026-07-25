# Puzzle-Joiner

Single-file (~2600 lines) Python/Qt desktop app for assembling puzzle pieces from scanned PDFs or images. Lives entirely in `puzzle-joiner`.

## Tech Stack

- **Python 3.11+**, packaged with `uv` (inline script dependencies)
- **PySide6** (Qt6) for UI
- **OpenCV** for image processing and feature matching
- **NumPy**, **tifffile** for array manipulation and TIFF export
- **Poppler** (`pdfseparate`, `pdftocairo`, `pdfinfo`) and **ImageMagick** (`convert`) for PDF handling

## Architecture (top to bottom in the file)

### Data Model (~line 51)

`PuzzlePiece` — holds a piece's full-res BGRA image, a display pixmap (max 2048px), a thumbnail (256px), and transform state (`x, y, rotation_deg, scale, crop_rect`). Key methods: `get_affine_matrix()` for the piece-local-to-world 2x3 matrix, `get_bounding_box()` for world-space bounds, `get_cropped_image()` for applying the crop rect.

### PDF Import & Preprocessing (~line 132)

Pipeline: `pdfseparate` splits PDF into pages -> ImageMagick converts to PNG at 300 DPI -> border line detection via Hough transform (`autocrop_border_lines`) -> legend/margin removal (`remove_legend`, `find_tall_verticals`, `find_legend_column`, `find_corner_boxes`) -> auto-crop whitespace. Results cached under `~/.cache/puzzle-joiner/{md5}/page_XXXX.png`.

### Feature Matching (~line 560)

- **ORB feature extraction** (`compute_orb_features`): extracts keypoints per color channel (B, G, R), masks out gray pixels (background). Uses WTA_K=4.
- **Pairwise matching** (`estimate_pairwise_transform`): BFMatcher with cross-check -> top 150 matches -> `cv2.estimateAffinePartial2D()`. Validates via overlap percentage and `matching_pixels_pct` (pixel-level verification within tolerance).
- **Auto-detect** (`auto_detect_placements`, ~line 711): greedy BFS starting from piece 0 at the origin. Iteratively matches unplaced pieces against placed ones (5-90% overlap, >75% pixel match). Unmatched pieces placed in a row below.
- **Snap** (`snap_piece_to_neighbors`, ~line 953): refines existing placement using 50k ORB features on the geometric overlap ROI between a piece and its neighbors.

### Export (~line 987)

- **Layered TIFF**: each piece warped to world coords as a separate TIFF layer (BigTIFF, deflate).
- **Flattened TIFF**: alpha-composite all pieces onto a single canvas.

### Threading (~line 1067)

`Worker(QRunnable)` + `WorkerSignals` pattern. Long tasks (PDF conversion, auto-detect, snap) run on QThreadPool with progress signals. PDF conversion uses `ProcessPoolExecutor`; image loading uses `ThreadPoolExecutor`.

### UI Components (~line 1097)

- **TransformHandles / HandleItem**: 13 draggable handles (8 corners, 4 edges, 1 rotation) for manual piece manipulation.
- **PuzzlePieceItem** (`QGraphicsPixmapItem`): syncs between model and scene. Red border when selected.
- **PuzzleScene**: maps `PuzzlePiece` -> `PuzzlePieceItem`, finds neighbors within margin for snapping.
- **PuzzleView**: mouse-wheel zoom, middle-click/space pan, fit-all.
- **ThumbnailPanel** (~line 1482): horizontal strip of piece thumbnails with lock/matched indicators and context menus.
- **PdfImportDialog** (~line 1810): page selection grid with async thumbnail generation.
- **CutToolDialog** (~line 2091): crop region editor.

### MainWindow (~line 2101)

Toolbar-driven workflow: Open PDF/Images -> Auto-Detect -> Snap (refine) -> Cut (crop) -> Lock -> Export. State is in-memory only (no save/load).

## Key Data Flow

```
PDF/Images -> convert & clean (parallel) -> cache -> PuzzlePiece list
  -> Auto-Detect (ORB matching, BFS placement) -> scene positions
  -> Manual adjust / Snap / Cut -> Export TIFF
```

## Running

```
uv run puzzle-joiner
```
