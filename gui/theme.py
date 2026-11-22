"""Uygulamanın renk teması.

Windows karanlık moddayken Qt (Fusion) karanlık palet seçer: yazılar beyaz olur.
Arayüzün stil dosyası ise arka planları açık renge sabitler; sonuç beyaz üstüne
beyaz yazı olur. Bu yüzden uygulama, sistem temasından bağımsız olarak her zaman
aynı açık paleti kullanır.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

_TEXT = "#0f172a"
_MUTED = "#64748b"
_DISABLED = "#94a3b8"


def light_palette() -> QPalette:
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: "#f8fafc",
        QPalette.ColorRole.WindowText: _TEXT,
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#f1f5f9",
        QPalette.ColorRole.Text: _TEXT,
        QPalette.ColorRole.Button: "#f8fafc",
        QPalette.ColorRole.ButtonText: _TEXT,
        QPalette.ColorRole.BrightText: "#dc2626",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: _TEXT,
        QPalette.ColorRole.PlaceholderText: _DISABLED,
        QPalette.ColorRole.Highlight: "#2563eb",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#2563eb",
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: "#e2e8f0",
        QPalette.ColorRole.Mid: "#cbd5e1",
        QPalette.ColorRole.Dark: "#94a3b8",
        QPalette.ColorRole.Shadow: "#475569",
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(_DISABLED))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#f1f5f9"))
    return p


def apply_light_theme(app: QApplication | None = None) -> None:
    """Fusion stili + sabit açık palet. Birden çok kez çağrılabilir."""
    app = app or QApplication.instance()
    if app is None:
        return
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)  # Qt 6.8+
    except AttributeError:
        pass
    app.setStyle("Fusion")
    app.setPalette(light_palette())
