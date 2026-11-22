"""İngilizce modda ekranda Türkçe metin kalıp kalmadığını bulan yardımcılar (testler için)."""

from __future__ import annotations

import re

# Türkçeye özgü harfler ve İngilizcede geçmeyen sık kelimeler
_TR_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TR_WORDS = re.compile(
    r"\b(ve|ile|bir|için|yok|seç|sil|eğit|yenile|ajan|bölüm|labirent|ayarlar|sonuç\w*|"
    r"başarı|adım|ödül|tekrar|etiket|durdur|oyna|hazır|çalışma|klasör\w*|varsayılan\w*|"
    r"oran|ortalama|ort\.|yöntem|kayıt|izle|hız|yeni|boyut\w*|grafik\w*|tablo\w*|rapor\w*)\b",
    re.IGNORECASE,
)

# İngilizce modda da olduğu gibi kalması gereken, çevrilmeyen değerler
ALLOWED = {"Türkçe", "sag_el", "ornek", "buyuk", "normal"}


def looks_turkish(text: str) -> bool:
    if not text or text.strip() in ALLOWED:
        return False
    stripped = text
    for word in ALLOWED:
        stripped = stripped.replace(word, "")
    return bool(_TR_CHARS.search(stripped) or _TR_WORDS.search(stripped))


def widget_texts(root) -> list[tuple[str, str]]:
    """root altındaki görünür metinler: (nereden, metin) listesi."""
    from PySide6.QtWidgets import (
        QAbstractButton, QComboBox, QFormLayout, QGroupBox, QLabel, QLineEdit,
        QTableWidget, QTabWidget, QWidget,
    )

    out: list[tuple[str, str]] = []
    widgets = [root, *root.findChildren(QWidget)]
    for w in widgets:
        name = f"{type(w).__name__}:{w.objectName() or ''}"
        if w.toolTip():
            out.append((name + " tooltip", w.toolTip()))
        if isinstance(w, (QLabel, QAbstractButton)):
            out.append((name, w.text()))
        if isinstance(w, QGroupBox):
            out.append((name, w.title()))
        if isinstance(w, QLineEdit) and w.placeholderText():
            out.append((name + " placeholder", w.placeholderText()))
        if isinstance(w, QComboBox):
            out.extend((name + " item", w.itemText(i)) for i in range(w.count()))
        if isinstance(w, QTabWidget):
            out.extend((name + " tab", w.tabText(i)) for i in range(w.count()))
        if isinstance(w, QTableWidget):
            for c in range(w.columnCount()):
                it = w.horizontalHeaderItem(c)
                if it is not None:
                    out.append((name + " header", it.text()))
        lay = w.layout()
        if isinstance(lay, QFormLayout):
            for r in range(lay.rowCount()):
                item = lay.itemAt(r, QFormLayout.ItemRole.LabelRole)
                if item is not None and isinstance(item.widget(), QLabel):
                    out.append((name + " form", item.widget().text()))
    return out


def turkish_leftovers(root) -> list[str]:
    return sorted({f"{where}: {text!r}" for where, text in widget_texts(root) if looks_turkish(text)})


def figure_texts(fig) -> list[str]:
    """matplotlib Figure içindeki bütün metinler (başlık, eksen, lejant, işaret)."""
    from matplotlib.text import Text

    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def figure_turkish_leftovers(fig) -> list[str]:
    return sorted({t for t in figure_texts(fig) if looks_turkish(t)})
