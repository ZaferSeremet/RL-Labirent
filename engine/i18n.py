"""Arayüz ve çıktı dili (Türkçe / İngilizce). SÖZLEŞME: imzaları değiştirme.

Kullanım: kullanıcıya görünen her metin `tx("Türkçe", "English")` ile yazılır;
o anki dile göre biri döner. Türkçe metin kodda kalır, ayrı sözlük dosyası yok.

    from engine.i18n import tx
    QPushButton(tx("Eğit", "Train"))
    tx("{n} çalışma silindi", "{n} runs deleted").format(n=3)

Dil, metnin üretildiği anda okunur. Arayüz dil değişince sekmeleri baştan kurar
(gui/main_window.py), bu yüzden widget'larda ayrıca "retranslate" gerekmez.

Çevrilmeyenler: kod yorumları ve docstring'ler, sonuç klasörlerindeki veri
biçimi (CSV sütunları, ayarlar.json alanları, `results/<çalışma>/grafikler/`
klasör adı), ajan adları (`ornek`, `sarsa`...), test türü anahtarları
(`normal`, `buyuk`), yöntem anahtarları (`random`, `sag_el`).
"""

from __future__ import annotations

import json
from pathlib import Path

from .settings import ROOT

LANGS: tuple[str, ...] = ("tr", "en")
LANG_NAMES: dict[str, str] = {"tr": "Türkçe", "en": "English"}

# Kullanıcıya özel, git'e girmez (.gitignore)
USER_SETTINGS_PATH = ROOT / "kullanici_ayarlari.json"

_lang = "tr"


def get_language() -> str:
    return _lang


def set_language(lang: str, *, persist: bool = False) -> None:
    """Dili değiştirir. persist=True ise seçim kullanici_ayarlari.json'a yazılır."""
    global _lang
    if lang not in LANGS:
        raise ValueError(f"Dil {LANGS} içinden olmalı: {lang!r}")
    _lang = lang
    if persist:
        data = _read_user_settings()
        data["dil"] = lang
        try:
            USER_SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass  # yazılamazsa yalnız bu oturum için geçerli


def load_saved_language() -> str:
    """Kaydedilmiş dili döndürür (yoksa ya da bozuksa "tr"). Dili DEĞİŞTİRMEZ."""
    lang = _read_user_settings().get("dil")
    return lang if lang in LANGS else "tr"


def tx(tr: str, en: str) -> str:
    """O anki dile göre Türkçe ya da İngilizce metni döndürür."""
    return en if _lang == "en" else tr


def is_english() -> bool:
    return _lang == "en"


# ------------------------------------------------- ortak görünen adlar

_KIND_LABELS = {
    "normal": ("Normal Labirent", "Normal Maze"),
    "buyuk": ("Büyük Labirent", "Large Maze"),
}
_KIND_SHORT = {
    "normal": ("Normal", "Normal"),
    "buyuk": ("Büyük", "Large"),
}
_METHOD_LABELS = {
    "random": ("Random", "Random"),
    "sag_el": ("Sağ el", "Right-hand"),
}


def kind_label(kind: str) -> str:
    """Test türü başlığı: "normal" -> "Normal Labirent" / "Normal Maze"."""
    pair = _KIND_LABELS.get(kind)
    return tx(*pair) if pair else kind


def kind_short(kind: str) -> str:
    """Kısa test türü: "buyuk" -> "Büyük" / "Large"."""
    pair = _KIND_SHORT.get(kind)
    return tx(*pair) if pair else kind


def method_label(method: str) -> str:
    """Baseline adları çevrilir ("sag_el" -> "Right-hand"); ajan adları olduğu gibi kalır."""
    pair = _METHOD_LABELS.get(method)
    return tx(*pair) if pair else method


def _read_user_settings() -> dict:
    try:
        data = json.loads(USER_SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
