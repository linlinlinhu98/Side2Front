import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# High DPI support
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from src.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("侧脸转正脸图像编辑系统")

    # Global style
    app.setStyleSheet("""
        QMainWindow { background: #f5f5f5; }
        QStatusBar { font-size: 11px; color: #999; }
        QStatusBar::item { border: none; }
    """)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
