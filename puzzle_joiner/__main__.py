import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Puzzle Joiner")
    app.setApplicationVersion("1.0")
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
