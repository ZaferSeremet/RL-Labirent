"""Masaüstü ve durum çubuğu bildirimleri (gui/CONTRACT.md)."""

from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QStyle, QSystemTrayIcon


def app_icon(widget: QMainWindow | None = None) -> QIcon:
    """QStyle standart simgelerinden uygulama simgesi döndürür."""
    style = widget.style() if widget is not None else QApplication.style()
    if style is None:
        return QIcon()
    return style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)


class Notifier(QObject):
    """Sistem tepsisi varsa QSystemTrayIcon.showMessage, yoksa durum çubuğu mesajı gösterir."""

    def __init__(self, window: QMainWindow | None = None) -> None:
        super().__init__(window)
        self._window = window
        self._tray: QSystemTrayIcon | None = None
        self._setup_tray()

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(app_icon(self._window), self)
        tray.setToolTip("RL Labirent")
        tray.show()
        self._tray = tray

    def notify(
        self,
        title: str,
        message: str,
        *,
        error: bool = False,
        timeout_ms: int = 5000,
    ) -> None:
        """Kullanıcıya kısa bildirim gönderir; modal pencere açmaz."""
        full_text = f"{title}: {message}" if title else message
        if self._window is not None and self._window.statusBar() is not None:
            self._window.statusBar().showMessage(full_text, timeout_ms)

        if (
            self._tray is not None
            and QSystemTrayIcon.isSystemTrayAvailable()
            and QSystemTrayIcon.supportsMessages()
        ):
            icon = (
                QSystemTrayIcon.MessageIcon.Warning
                if error
                else QSystemTrayIcon.MessageIcon.Information
            )
            self._tray.showMessage(title, message, icon, timeout_ms)

    def cleanup(self) -> None:
        """Pencere kapanırken sistem tepsisi simgesini temizler."""
        if self._tray is not None:
            self._tray.hide()
            self._tray = None
