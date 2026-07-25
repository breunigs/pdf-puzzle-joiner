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
    args = sys.argv[1:]
    if args:
        # Check if PDF
        if len(args) == 1 and args[0].lower().endswith(".pdf"):
            QTimer.singleShot(100, lambda: window._import_pdf(args[0], None))
        else:
            # Treat as images
            QTimer.singleShot(100, lambda: window._import_images(args))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
