import logging
import os
import shutil
import sys

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer

from .cache import cleanup_old_cache_entries
from .main_window import MainWindow

# External binary -> apt package name
_REQUIRED_EXTERNALS = {
    "pdfseparate": "poppler-utils",
    "pdftocairo": "poppler-utils",
    "pdfinfo": "poppler-utils",
    "convert": "imagemagick",
}


def _check_external_dependencies():
    """Check that all required external binaries are available.

    Returns None if all present, or an error message string listing
    what is missing with an apt-get install command.
    """
    missing_packages = {}
    for binary, package in _REQUIRED_EXTERNALS.items():
        if not shutil.which(binary):
            missing_packages.setdefault(package, []).append(binary)
    if not missing_packages:
        return None
    lines = ["The following required external tools are missing:\n"]
    for package, binaries in sorted(missing_packages.items()):
        lines.append(f"  • {', '.join(binaries)}  (from {package})")
    pkg_list = " ".join(sorted(missing_packages))
    lines.append(f"\nInstall them with:\n\n  sudo apt-get install {pkg_list}")
    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cleanup_old_cache_entries()
    app = QApplication(sys.argv)
    app.setApplicationName("Puzzle Joiner")
    app.setApplicationVersion("1.0")

    dep_error = _check_external_dependencies()
    if dep_error:
        QMessageBox.critical(None, "Missing Dependencies", dep_error)
        sys.exit(1)

    window = MainWindow()
    window.show()

    # If files passed on command line, load them
    args = [os.path.abspath(a) for a in sys.argv[1:]]
    if args:
        window._input_path = args[0]
        pdfs = [a for a in args if a.lower().endswith(".pdf")]
        images = [a for a in args if not a.lower().endswith(".pdf")]
        def load_cli_files():
            if pdfs:
                window._open_pdfs(pdfs)
            if images:
                window._import_images(images)
        QTimer.singleShot(100, load_cli_files)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
