"""RL Labirent masaüstü uygulaması giriş noktası."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from engine import i18n
    from gui.main_window import MainWindow
    from gui.notify import app_icon
    from gui.theme import apply_light_theme

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    i18n.set_language(i18n.load_saved_language())  # son seçilen dil
    app.setApplicationName(i18n.tx("RL Labirent", "RL Maze"))
    apply_light_theme(app)  # Windows karanlık modda da okunaklı kalsın
    app.setWindowIcon(app_icon())

    window = MainWindow(persist_language=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
