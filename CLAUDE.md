# Puzzle-Joiner

Python/Qt desktop app for assembling puzzle pieces from scanned PDFs or images. Split into a `puzzle_joiner/` package, launched via `./puzzle-joiner` (uv script).

## Tech Stack

- **Python 3.14+**, packaged with `uv` (inline script dependencies in `puzzle-joiner`)
- **PySide6** (Qt6) for UI
- **OpenCV** for image processing and feature matching
- **NumPy**, **tifffile** for array manipulation and TIFF export
- **Poppler** (`pdfseparate`, `pdftocairo`, `pdfinfo`) and **ImageMagick** (`convert`) for PDF handling
- **System QGIS** (Qt5/PyQt5, optional) for OSM tiles and GCP fitting. The georeferenced export needs no GDAL — it writes the GeoTIFF tags itself. Both are system packages, not wheels; see `georef.py` for how a Qt5 library runs inside this Qt6 app.

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
  legend.py                # Legend instances, export anchor + uprighting math
  georef.py                # QGIS bridge: OSM tile rendering + GCP transform fitting
  export.py                # Layered + flattened TIFF export
  undo.py                  # QUndoCommand subclasses (TransformCommand, SnapCommand, CropCommand, LockCommand, GeorefCommand)
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
  test_lasso.py            # lasso crop: alpha masking, polygon editor, cut-tool undo
  test_legend.py           # legends: cutting, anchor/upright math, gating, persistence
  test_export.py           # export: rotation, band streaming, lasso + uncropped canvas
  test_georef.py           # georeferencing: canvas GCP mapping, fit, persistence, dialog
  test_boot.py             # smoke test: MainWindow constructs, legend add/undo/restore
```

## Module Responsibilities

### `model.py`
`PuzzlePiece` — holds full-res BGRA image, display pixmap (max 2048px), thumbnail (256px), and transform state (`x, y, rotation_deg, scale, crop_rect`, `lasso_poly`). Key methods: `get_affine_matrix()`, `get_bounding_box()`, `get_cropped_image()`.

`gcps` holds the page's ground control points as `[(img_x, img_y, map_x, map_y), ...]`. The image coords are **full-image** ones — the same space as `crop_rect` and `lasso_poly`, and the space `GeorefDialog` shows — so a point stays put when the crop changes. See `georef.py`.

`layout_key()` is the piece's identity for layout persistence. Pages key on their source basename; legend instances share their page's source path, so they append `#legend{n}` to be told apart from the page and from each other. See `legend.py` for the rest of the legend state.

**Cropping.** `crop_rect` is always the rectangle that defines the piece's local coordinate system, so all geometry stays rect-based. An optional `lasso_poly` (polygon in full-image coords, with `crop_rect` holding its bounding box) additionally masks the alpha channel, which is what hides the outside in the preview and the export. `apply_crop(crop_rect, lasso_poly)` sets both and refreshes the mask; the mask is written into `image`'s alpha in place so `get_cropped_view()` stays zero-copy, with the original channel kept in `_base_alpha` so edits are repeatable. `get_match_image()` returns the rect crop with the mask lifted — matching deliberately ignores the lasso so a rough outline cannot cost a pair its overlap. `get_outline_points()` returns the opaque extent in local coords, which the export uses to size its canvas.

### `preprocessing.py`
Pipeline: `pdfseparate` → ImageMagick convert at 300 DPI → border line detection via Hough transform (`autocrop_border_lines`) → legend/margin removal (`remove_legend`) → auto-crop computation (`compute_auto_crop_rect`). Also contains `_convert_and_process_page` worker for ProcessPoolExecutor. Imports `LOW_PRIO` from `priority`.

### `cache.py`
Persistent cache at `~/.cache/puzzle-joiner/{md5}/`. Stores uncropped page PNGs (`page_XXXX.png`) and auto-crop rects (`page_XXXX_autocrop.json`). `_load_cached_page` reads or recomputes crop rect on load. Imports `compute_auto_crop_rect` from `preprocessing`.

### `matching.py`
- **ORB extraction** (`compute_orb_features`): per-channel (B,G,R), optional gray masking, WTA_K=4.
- **Pairwise matching** (`estimate_pairwise_transform`): BFMatcher cross-check → `estimateAffinePartial2D`. Pixel-level validation.
- **Auto-detect** (`auto_detect_placements`): greedy BFS placement, >75% pixel match threshold.
- **Snap** (`snap_piece_to_neighbors`): 50k features, no gray masking, ROI-based overlap matching, ≥50% pixel threshold.

### `legend.py`
A legend is a region lifted off a page and placed beside the assembled map — a `PuzzlePiece` with `is_legend` set, owning its own copy of the pixels (so editing the page cannot punch holes in it, and one region can be included any number of times). `make_legend_piece` cuts one from a page, taking `_base_alpha` so a lasso drawn round the map does not make the legend transparent; `clone_legend_piece` duplicates an instance.

**Anchor and uprighting.** On the canvas a legend rotates freely, which is how a user lines it up against a map sitting at an arbitrary angle. The export throws that rotation away: `anchor_local` finds the point of the legend nearest the map (`map_hull`, densified outline so the winner can land mid-edge), and `upright_affine` re-emits the legend axis-aligned about that point. Pivoting on the anchor is what keeps the legend as close to the map as the user placed it, whatever rotation the export applies to the map. Distances are measured in canvas space — the global rotation is rigid, so the winning point is the same one in export space. `anchor_pair` returns the world-space anchor and its nearest map point, which the canvas draws as the export-pivot indicator.

Note the legend body *does* swing when it is uprighted; only the anchor is pinned. The pivot is shown on the canvas so that swing is predictable.

**What a legend is measured against.** `map_outlines` gives each map piece's outline in world coords — outlines, not hulls. One hull over the whole map fills in every bend of the road it follows, and even a hull per page measured two to three times the ground that page covers. Legends pushed clear of a shape that size finished up to 120 m from the map, having been moved as much as 250 m to get there; against the real outlines they move 0–11 m and sit 0.3–15 m clear. `convex_parts` ear-clips an outline into triangles for the ray clipping, which needs convex shapes — together they cover the outline exactly, so nothing is fattened to buy that convexity. 256 triangles for a five-page plan, 6 ms.

**Separation.** Because the swing changes a legend's footprint, a legend clear of everything on the canvas can land on the map or on a neighbour once squared up — the pivots are per legend, so nothing about the rotation keeps two of them apart. `separate_legends` resolves that afterwards: legends are processed nearest-the-map first (ties by instance number, so the result does not depend on list order), and each is translated the shortest distance that clears the map and every legend already placed. Candidate directions are straight out from the map plus both sideways; reversing back towards the map is only allowed for a legend that *starts* on top of it, where the nearest edge is the way out. The distance along a direction is exact, not iterative: `moving + t*d` meets an obstacle exactly when `t*d` is inside the Minkowski difference, which is convex, so `_ray_overlap_interval` gets the overlapping `t` range by clipping the ray against its edges, and `_clear_distance` sweeps those ranges until `t` is outside all of them. A nudge is a pure translation, so it cannot reintroduce rotation.

### `georef.py`
Borrows the **system QGIS** install (`/usr/lib/python3/dist-packages`, appended to `sys.path` so the app's own numpy still wins). QGIS is a package, not a wheel, so it may be absent — `available()` is the gate every caller checks, and the Georeference action just reports it is missing.

**Two Qt versions, one process.** QGIS is Qt5/PyQt5 and this app is Qt6/PySide6. They coexist because Qt versions its symbols, on one condition: `QgsApplication([], False)` — GUI disabled, so QGIS never loads a Qt5 platform plugin and never contests the event loop. PySide6 owns all the windowing. Hence `qgis.gui` is unusable and nothing here returns a Qt object: pixels leave as numpy arrays, points as tuples, so a Qt5 wrapper can never reach a PySide6 call. `initialize()` must run on the GUI thread; `ready()` guards the renders, which would crash rather than fail without it. `render_osm` is safe off-thread — the job runs its own threads and `waitForFinished` needs no event loop of ours.

What is borrowed is the tile stack (fetching, caching, zoom choice, reprojection, labels) and `QgsGcpTransformerInterface`. Map coordinates are EPSG:3857 throughout, which is what the tiles render in and what the user's clicks are measured in, so the warp reprojects nothing.

**The row flip.** Image rows run down and northings run up, so the mapping contains a reflection, which a similarity transform cannot express — given raw rows, Helmert approximates the flip by rotating the map 45°. QGIS wants the reflection taken out of the points it is *fitted* on but puts it back itself when *transforming*, so `fit()` negates the row on input and not on query. That asymmetry was measured, not deduced: of the four sign combinations only this one reproduces a known north-up mapping exactly (`test_georef.TestFit`). Linear survives either way — having no rotation to protect, it absorbs the flip as a negative y scale.

### `export.py`
- **Layered TIFF**: each piece as a separate BigTIFF layer (deflate, 1024x1024 tiles).
- **Flattened TIFF**: alpha-composite all pieces onto single canvas.
- **Uncropped flattened TIFF** (`export_flattened_tiff(..., uncropped=True)`): the same composite with every lasso reduced to its bounding rect, written alongside the others as `{name}_uncropped_flat.tif`. `_lassos_lifted` puts each piece's `_base_alpha` back for the duration of the write rather than copying pixels, so the warps stay zero-copy, and `refresh_lasso_mask()` rebuilds the mask afterwards.
- **One canvas for every file.** `export_canvas` is sized from the crop rects — never from the lasso hulls, which would clip the uncropped file — so all three files come out the same size on the same origin and overlay exactly. That is the point of the uncropped file: georeference it, then swap the lassoed one in for display. Only the interpolation differs, by at most one step, because the two renders hand `warpAffine` different box offsets.
- `_warp_region` fills the destination a column at a time and crops the source to the window that can actually reach it. Both halves matter at this size: the whole page was otherwise handed to OpenCV for every band it touches, and OpenCV's **Lanczos path refuses a source at or past SHRT_MAX on either axis** — which one page of a 300 dpi plan reaches on its own. Linear and cubic have no such limit, so cubic is the fallback if a source slice ever exceeds it anyway. Rows are tiled as well as columns, because a band on a narrow canvas runs to tens of thousands of rows and would push the source window past the limit by its height alone. Re-origining shifts the fixed-point sample position by up to 1/32 px, so output is not bit-identical to one huge call: bilinear stays within a level, Lanczos reaches about a dozen on the odd pixel of hard linework.
- Per piece, `_piece_bbox` still renders only the area that can show, so a lasso keeps its warp down to the polygon hull. The hull is padded by the piece's scale plus 2 px: `fillPoly` antialiases the mask outside the polygon and the warp spreads that again, and the same pad on the rect box keeps it a superset so the shared canvas clips neither file.
- **Georeferenced TIFF** (`{name}_warped.tif`): offered whenever a page carries control points. `export_geotiff` folds the georeferencing into the affines the compositor already applies, so there is **no warping step and no intermediate raster** — the pixels are read once, resampled once (Lanczos), and streamed band by band like every other export. That is what lets it survive a canvas far larger than memory; a 5.8 Gpx export of a real plan takes ~130 s and 221 MB.
- `canvas_gcps` pools every page's points in canvas coordinates, with no origin subtracted: the export derives its own origin after transforming. The coordinates are absolute, so adding or removing a legend cannot shift them.
- `georef_transform` turns the fit into a grid whose pixel **is the fit's own scale**. At that pixel size the map's transform is a pure rotation, which is the whole trick: no rescaling of anything, and `_export_affines(pieces, global_transform=G)` then handles legends with the machinery that already existed. A legend rides the georeferencing as far as its anchor and is re-emitted upright there, so it **follows the map round without being turned with it** — see `legend.py`. `export_order` still composites legends last, so they sit on top. Verified on a plan whose pages sit at 16.6°: the legends come out axis-aligned and crisp.
- **Map footprint** (`{name}_warped.geojson`): `map_footprint` traces the ground the map covers from the map pieces alone, so a legend — which sits beside the map, over ground it has nothing to do with — is never part of it. It cannot be read back off the finished raster: nothing in those pixels says which blob is a legend, and on a real plan `gdal_footprint` returned seven parts, six of them legends, with the map **fourth** in the list. Sorting by size works until a small map meets a large legend, so the export states the answer instead of leaving it to be inferred. `write_qgis_project` carries it as a vector layer, where the task's `.tif` filter passes over it.
- `_geotiff_tags` writes the pixel scale, tie point and CRS straight into the TIFF. Bolting them on afterwards with `gdal_translate` would re-encode every tile to record six numbers.
- `_compute_optimal_rotation` sees map pieces only: a legend is not part of the shape being minimised, and it does not keep the angle it is measured at. `_export_affines` then rides each legend along the global rotation as far as its anchor, re-emits it upright, and runs `separate_legends` to nudge any overlaps apart (all three in `legend.py`); everything downstream is affine-generic and needs no legend case. `export_order` puts legends last so they composite on top.

### `widgets.py`
- `HandleItem`, `TransformHandles`: GIMP-style resize/rotate handles.
- `PuzzlePieceItem`: syncs between model and scene, red border when selected.
- `PuzzleScene`, `PuzzleView`: virtual canvas with zoom/pan.
- `ThumbnailPanel`: horizontal strip with lock/matched indicators; green border for legends, which are never "matched" and so must skip the unmatched tint. Right-click offers Lock plus Add Legend on a page, Duplicate/Delete on a legend, relayed to `MainWindow` as signals.
- `PuzzleScene.update_legend_anchor`: dashed marker showing the point the export will pivot a selected legend on. `get_neighbors` and `get_drag_group` both exclude legends, so no snap can reach them and no lock group can drag them.
- `PdfImportDialog`: page selection grid with async thumbnail generation.
- `CutToolDialog`: crop region editor showing the full uncropped image, unmasked so an existing lasso can be pushed outwards. Two tools: Rectangle (`CropRectItem`) and Lasso (`LassoItem`).
- `ExportOptionsDialog`: checkboxes for which files an export writes, remembered for the next export in the same session. Every entry stands alone — the georeferenced file composites the pieces itself rather than warping the flat one, so it needs nothing but control points, and any single box is a valid export. It is also the sole default once the pages are georeferenced; without control points it cannot be written at all and the flat file takes over, so the dialog never opens with nothing to agree to.
- `GeorefDialog`: page beside OSM, click one then the other to pair a control point. The page is shown **uncropped** like the cut tool (its crop outlined dashed, as a reminder of what survives), so a landmark just outside the crop is still usable. The splitter follows the page's long axis — a tall page leaves width to spare so the panes go side by side, a wide one stacks them — and the toggle flips it. Transform choice (Helmert default, Linear) lives here and is saved per document. Both panes are fitted in `showEvent`, not the constructor, where the widgets still have provisional geometry.
- `GcpView` / `OsmView` / `GcpMarkerItem`: shared pan/zoom (middle-drag, wheel) so both halves of a pair are placed the same way. `OsmView` scene coords are map metres with the **northing negated**, so scene y runs down like every other view and like the rendered rows; `scene_to_map` is therefore the only conversion the dialog needs. Tiles are fetched for the visible rect only, debounced, on a worker thread, each request carrying a token so a slow one landing late cannot overwrite a newer one. Refresh is driven off the **scrollbars**, not the gestures, so any scroll route (keys, trackpad) is covered. Markers ignore the view transform, since the map pane ranges from the whole world to one building.
- `LassoItem`, `LassoVertexItem`: GIMP-style piecewise-linear selection — click to place points, close by clicking the first point / double-click / Enter, Backspace to undo a point, drag a vertex to correct it, drag a midpoint knob to insert one, right-click to delete. `CropView` owns all the hit-testing and dragging, by index: moving a point can rebuild the handle list, so no handle may hold the mouse grab.

### `main_window.py`
Toolbar workflow: Open PDF/Images → Auto-Detect → Snap → Cut → Add Legend → Georeference → Lock → Export. Manages piece list, threading, progress dialogs.

**Export.** `on_export` opens `ExportOptionsDialog` first and asks for a path second, so backing out of the choice costs nothing. The selected files are written one after another, never concurrently: each sizes its band to the free memory, and two in flight would each believe they had the RAM the other was using.

**Georeferencing.** `on_georeference` opens `GeorefDialog` on the selected page (legends carry no geography and are refused) and pushes a `GeorefCommand`; control points draw nothing on the canvas, so it needs no `sync_fn`. The method is one per document, kept in `self._georef_method`, saved into `layout.json` — whose cache dir is already per input PDF — and read back by `load_georef_method`. `on_export` starts QGIS on the GUI thread before the export worker begins, since the fit runs on that worker.

**Legends.** `_map_pieces()` is the matching set — Auto-Detect and Snap both run on it, so legends are never matched or moved by them (Auto-Detect leaves them exactly where they are and says so in the status bar). `add_legend_from_page` opens `CutToolDialog` in `mode="legend"`, `duplicate_legend` clones one, and both go on the undo stack via `AddPieceCommand`. `_restore_legends` re-cuts saved legends before `load_layout` positions them, skipping any whose page is not loaded.

### `priority.py`
Builds `LOW_PRIO` prefix (`chrt --idle 0 nice -n19 ionice --class idle`) for subprocesses. `_set_low_priority()` for ProcessPoolExecutor worker initializer.

**Pool workers start cold.** Python 3.14 defaults to the `forkserver` start method on Linux, so a pool worker is a fresh interpreter rather than a copy of the parent: it re-imports the module and inherits no runtime state. Both pool entry points (`_load_cached_page`, `_convert_and_process_page`) and the initializer must therefore stay module-level and picklable, and must take everything they need through their args tuple — a module global set after import, or a closure, will not reach them. `LOW_PRIO` is safe because it is computed at import time, so the worker rebuilds it. This also means the pool no longer forks a live Qt process, which `fork` did.

## Key Data Flow

```
PDF/Images → convert & clean (parallel, low-prio) → cache → PuzzlePiece list
  → Auto-Detect (ORB matching, BFS placement) → scene positions
  → Manual adjust / Snap / Cut → Export TIFF
                              ↘ Add Legend (region off any page, excluded from
                                matching) → placed by hand → uprighted on export
                              ↘ Georeference (per page, against OSM) → points
                                pooled → fit folded into the export affines →
                                {name}_warped.tif (+ .qgs), legends upright
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
