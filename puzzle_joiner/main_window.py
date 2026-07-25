import os
import sys
import tempfile
import concurrent.futures
from typing import Optional

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QStatusBar,
    QFileDialog, QDialog, QMessageBox, QProgressDialog,
)
from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtGui import QAction

from .model import PuzzlePiece
from .preprocessing import (
    parse_page_range, split_pdf_pages, pdf_page_info,
    _convert_and_process_page,
)
from .cache import (
    _cache_dir_for_pdf, _cached_page_path, _cached_autocrop_path,
    _load_cached_page,
)
from .matching import auto_detect_placements, snap_piece_to_neighbors
from .export import export_layered_tiff, export_flattened_tiff
from .worker import Worker
from .priority import _set_low_priority
from .widgets import (
    PuzzleScene, PuzzleView, ThumbnailPanel,
    PdfImportDialog, CutToolDialog,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Puzzle Joiner")
        self.resize(1400, 900)
        self.pieces: list = []
        self._work_dir = None
        self._thread_pool = QThreadPool.globalInstance()

        self._build_ui()

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        self.action_open_pdf = QAction("Open PDF", self)
        self.action_open_pdf.triggered.connect(self.on_open_pdf)
        toolbar.addAction(self.action_open_pdf)

        self.action_open_images = QAction("Open Images", self)
        self.action_open_images.triggered.connect(self.on_open_images)
        toolbar.addAction(self.action_open_images)

        toolbar.addSeparator()

        self.action_auto_detect = QAction("Auto-Detect", self)
        self.action_auto_detect.triggered.connect(self.on_auto_detect)
        toolbar.addAction(self.action_auto_detect)

        self.action_snap = QAction("Snap", self)
        self.action_snap.triggered.connect(self.on_snap)
        toolbar.addAction(self.action_snap)

        toolbar.addSeparator()

        self.action_cut = QAction("Cut", self)
        self.action_cut.triggered.connect(self.on_cut)
        toolbar.addAction(self.action_cut)

        self.action_lock = QAction("Lock/Unlock", self)
        self.action_lock.triggered.connect(self.on_lock_toggle)
        toolbar.addAction(self.action_lock)

        toolbar.addSeparator()

        self.action_export_layered = QAction("Export Layered TIFF", self)
        self.action_export_layered.triggered.connect(self.on_export_layered)
        toolbar.addAction(self.action_export_layered)

        self.action_export_flat = QAction("Export Flat TIFF", self)
        self.action_export_flat.triggered.connect(self.on_export_flat)
        toolbar.addAction(self.action_export_flat)

        toolbar.addSeparator()

        self.action_fit = QAction("Fit View", self)
        self.action_fit.triggered.connect(self.on_fit_view)
        toolbar.addAction(self.action_fit)

        # Thumbnail panel
        self.thumbnail_panel = ThumbnailPanel()
        self.thumbnail_panel.piece_selected.connect(self.on_thumbnail_select)
        main_layout.addWidget(self.thumbnail_panel)

        # Canvas
        self.scene = PuzzleScene()
        self.scene.selection_changed_signal.connect(self.on_scene_selection_changed)
        self.scene.piece_moved_signal.connect(self._update_snap_enabled)
        self.view = PuzzleView(self.scene)
        main_layout.addWidget(self.view, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open a PDF or images to begin.")

    def _run_with_progress(self, title: str, fn, *args, on_done=None, **kwargs):
        """Run fn(*args, **kwargs, progress=Signal) in background with QProgressDialog."""
        dlg = QProgressDialog(title, "Cancel", 0, 100, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        worker = Worker(fn, *args, **kwargs)

        def on_progress(pct, msg):
            dlg.setValue(pct)
            dlg.setLabelText(msg)

        def on_finished(result):
            dlg.setValue(100)
            dlg.close()
            if on_done:
                on_done(result)

        def on_error(msg):
            dlg.close()
            QMessageBox.critical(self, "Error", msg)

        worker.signals.progress.connect(on_progress)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)

        self._thread_pool.start(worker)
        dlg.exec()

    # --- PDF / Image loading ---

    def on_open_pdf(self):
        pdf_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF files (*.pdf)"
        )
        if not pdf_path:
            return

        page_count, page_sizes = pdf_page_info(pdf_path)
        dlg = PdfImportDialog(pdf_path, page_count, page_sizes, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        pages = dlg.get_pages()
        self._import_pdf(pdf_path, pages)

    def _import_pdf(self, pdf_path: str, pages):
        if self._work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix="puzzle_joiner_")

        def task(progress):
            progress.emit(2, "Computing cache key...")
            cache_dir = _cache_dir_for_pdf(pdf_path)

            # Determine which pages are requested (1-indexed)
            if pages:
                requested = sorted(pages)
            else:
                page_count, _ = pdf_page_info(pdf_path)
                requested = list(range(1, page_count + 1))

            # Check cache for each requested page
            cached_pages = {}
            uncached_page_nums = []
            for page_num in requested:
                cp = _cached_page_path(cache_dir, page_num)
                if os.path.exists(cp) and os.path.getsize(cp) > 0:
                    cached_pages[page_num] = cp
                else:
                    uncached_page_nums.append(page_num)

            results = []
            total = len(requested)

            # Load cached pages in parallel (compute crop rect on load)
            if cached_pages:
                load_args = [(pn, cached_pages[pn], requested.index(pn))
                             for pn in requested if pn in cached_pages]
                workers = max(1, os.cpu_count() // 2)
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_set_low_priority) as executor:
                    for i, res in enumerate(executor.map(_load_cached_page, load_args)):
                        if res is not None:
                            results.append(res)
                        progress.emit(5 + int(40 * (i + 1) / len(load_args)),
                                      f"Loading cached page {i + 1}/{len(load_args)}...")

            if uncached_page_nums:
                # Split PDF to get per-page PDFs, only for uncached pages
                progress.emit(45, "Splitting PDF...")
                page_pdfs = split_pdf_pages(pdf_path, self._work_dir, set(uncached_page_nums))
                workers = max(1, os.cpu_count() // 2)
                n_uncached = len(page_pdfs)

                # Build args for convert+process workers
                convert_args = []
                for pdf_page_num, pdf_page_path in page_pdfs:
                    idx = requested.index(pdf_page_num)
                    cp = _cached_page_path(cache_dir, pdf_page_num)
                    ac_path = _cached_autocrop_path(cache_dir, pdf_page_num)
                    convert_args.append((pdf_page_path, pdf_page_num, idx, cp, ac_path))

                completed = 0
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_set_low_priority) as executor:
                    futures = {executor.submit(_convert_and_process_page, a): a[2] for a in convert_args}
                    for future in concurrent.futures.as_completed(futures):
                        completed += 1
                        pct = 50 + int(45 * completed / n_uncached)
                        progress.emit(pct, f"Processing page {completed}/{n_uncached}...")
                        res = future.result()
                        if res is not None:
                            results.append(res)

            # Sort by page index (idx is last element)
            results.sort(key=lambda r: r[-1])
            progress.emit(98, "Done.")
            return results

        def on_done(result):
            if result is None:
                return
            for (path, bgra, crop_rect, idx) in result:
                self._add_piece_from_array(bgra, path, idx, crop_rect=crop_rect)
            n = len(result)
            name = os.path.basename(pdf_path)
            self.status_bar.showMessage(f"Loaded {n} pages from {name}")
            self.view.fit_all()

        self._run_with_progress("Importing PDF", task, on_done=on_done)

    def on_open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Images", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
        )
        if not paths:
            return
        self._import_images(paths)

    def _import_images(self, paths: list):
        def task(progress):
            result = []
            for idx, path in enumerate(paths):
                pct = int(90 * idx / max(len(paths), 1))
                progress.emit(pct, f"Loading {os.path.basename(path)}...")
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                result.append((path, img, idx))
            progress.emit(95, "Done.")
            return result

        def on_done(result):
            if not result:
                return
            for (path, bgra, idx) in result:
                self._add_piece_from_array(bgra, path, idx)
            self.status_bar.showMessage(f"Loaded {len(result)} images")
            self.view.fit_all()

        self._run_with_progress("Loading images", task, on_done=on_done)

    def _add_piece_from_array(self, bgra: np.ndarray, source_path: str, idx: int,
                              crop_rect: tuple = None):
        piece = PuzzlePiece()
        piece.source_path = source_path
        piece.image = bgra
        piece.layer_index = idx
        piece.crop_rect = crop_rect
        img = piece.get_cropped_image()
        h, w = img.shape[:2]
        # Default position: stacked horizontally
        piece.x = w / 2.0 + idx * (w + 20)
        piece.y = h / 2.0
        piece.rebuild_display_pixmap()
        piece.rebuild_thumbnail()
        self.pieces.append(piece)
        self.scene.add_piece(piece)
        self.thumbnail_panel.add_piece(piece)

    # --- Auto-detect ---

    def on_auto_detect(self):
        if not self.pieces:
            self.status_bar.showMessage("No pieces loaded.")
            return

        def task(progress):
            progress.emit(10, "Computing ORB features...")
            auto_detect_placements(self.pieces)
            progress.emit(90, "Syncing positions...")
            return True

        def on_done(result):
            self.scene.sync_all_from_pieces()
            self.thumbnail_panel.update_all()
            self.view.fit_all()
            self._update_snap_enabled()
            self.status_bar.showMessage("Auto-detect complete.")

        self._run_with_progress("Auto-detecting placements", task, on_done=on_done)

    # --- Snap ---

    def on_snap(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            # Snap all unmatched
            self._snap_all()
            return
        neighbors = self.scene.get_neighbors(piece)
        if not neighbors:
            self.status_bar.showMessage("No neighbors found for snapping.")
            return

        def task(progress):
            progress.emit(20, "Computing features...")
            ok = snap_piece_to_neighbors(piece, neighbors)
            progress.emit(90, "Done.")
            return ok

        def on_done(ok):
            item = self.scene.piece_items.get(piece)
            if item:
                item.sync_from_piece()
            if ok:
                piece.is_matched = True
                self.thumbnail_panel.update_piece(piece)
                self.status_bar.showMessage("Snap successful.")
            else:
                self.status_bar.showMessage("Snap found no match.")
            self._update_snap_enabled()

        self._run_with_progress("Snapping piece", task, on_done=on_done)

    def _snap_all(self):
        def task(progress):
            total = len(self.pieces)
            for i, piece in enumerate(self.pieces):
                progress.emit(int(90 * i / max(total, 1)), f"Snapping piece {i+1}/{total}...")
                neighbors = self.scene.get_neighbors(piece)
                if neighbors:
                    ok = snap_piece_to_neighbors(piece, neighbors)
                    if ok:
                        piece.is_matched = True
            return True

        def on_done(result):
            self.scene.sync_all_from_pieces()
            self.thumbnail_panel.update_all()
            self._update_snap_enabled()
            self.status_bar.showMessage("Snap all complete.")

        self._run_with_progress("Snapping all pieces", task, on_done=on_done)

    # --- Cut tool ---

    def on_cut(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self.status_bar.showMessage("Select a piece first.")
            return
        dlg = CutToolDialog(piece, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            item = self.scene.piece_items.get(piece)
            if item:
                item.rebuild_pixmap()
            piece.rebuild_thumbnail()
            self.thumbnail_panel.update_piece(piece)
            self.status_bar.showMessage("Crop applied.")

    # --- Lock/Unlock ---

    def on_lock_toggle(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self.status_bar.showMessage("Select a piece first.")
            return
        piece.is_locked = not piece.is_locked
        item = self.scene.piece_items.get(piece)
        if item:
            item.sync_from_piece()
            if piece.is_locked:
                item.setSelected(False)
        self.thumbnail_panel.update_piece(piece)
        state = "locked" if piece.is_locked else "unlocked"
        self.status_bar.showMessage(f"Piece {state}.")

    # --- Export ---

    def on_export_layered(self):
        if not self.pieces:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Layered TIFF", "puzzle_layered.tif", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        def task(progress):
            progress.emit(10, "Exporting layered TIFF...")
            export_layered_tiff(self.pieces, path)
            progress.emit(95, "Done.")
            return path

        def on_done(out_path):
            self.status_bar.showMessage(f"Exported: {out_path}")

        self._run_with_progress("Exporting", task, on_done=on_done)

    def on_export_flat(self):
        if not self.pieces:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Flattened TIFF", "puzzle_flat.tif", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        def task(progress):
            progress.emit(10, "Exporting flattened TIFF...")
            export_flattened_tiff(self.pieces, path)
            progress.emit(95, "Done.")
            return path

        def on_done(out_path):
            self.status_bar.showMessage(f"Exported: {out_path}")

        self._run_with_progress("Exporting", task, on_done=on_done)

    # --- Fit View ---

    def on_fit_view(self):
        self.view.fit_all()

    # --- Selection sync ---

    def _update_snap_enabled(self):
        piece = self.scene.get_selected_piece()
        if piece is None:
            self.action_snap.setEnabled(False)
        else:
            self.action_snap.setEnabled(bool(self.scene.get_neighbors(piece)))

    def on_scene_selection_changed(self, piece):
        self.thumbnail_panel.select_piece(piece)
        self._update_snap_enabled()

    def on_thumbnail_select(self, piece: PuzzlePiece):
        # Clear current selection
        self.scene.clearSelection()
        item = self.scene.piece_items.get(piece)
        if item:
            item.setSelected(True)
            # Center view on piece
            bb = piece.get_bounding_box()
            self.view.centerOn(bb.center().x(), bb.center().y())
        self.thumbnail_panel.select_piece(piece)
        self._update_snap_enabled()
