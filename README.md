# RL Labirent

Derin Pekiştirmeli Öğrenme dersi grup ödevi için ortak ortam ve eğitim/test uygulaması.
Herkes kendi algoritmasını yazar; ortam, eğitim, test, grafikler ve rapor herkes için aynıdır.

**Problem:** Ajan her bölümde rastgele üretilen bir labirentte çıkışı arar. Labirentin tamamını,
kendi koordinatını ve çıkışın yerini bilmez; yalnız çevresindeki küçük pencereyi (varsayılan 3x3),
o penceredeki karelere daha önce girip girmediğini ve bir önceki hamlesini görür.
Ayrıntılı tanım: [maze_env/SPEC.md](maze_env/SPEC.md).

**İçindekiler:**
1. [Kurulum](#1-kurulum)
2. [Uygulamayı açma](#2-uygulamayı-açma)
3. [Kendi ajanını yazma](#3-kendi-ajanını-yazma)
4. [Deney akışı](#4-deney-akışı)
5. [Sonuçları gönderme](#5-sonuçları-gönderme)
6. [Kurallar](#6-kurallar)
7. [Sorun giderme](#7-sorun-giderme)

---

## 1. Kurulum

Gerekenler: **Python 3.11 ya da daha yeni** (3.11 ve 3.14'te denendi). Kurulum ilk seferde ~200 MB paket indirir.

Kod sana `RL-Labirent_<tarih>_<sürüm>.zip` olarak gelir. Zip'i istediğin yere çıkar; içinden çıkan `RL-Labirent` klasörü proje klasörüdür. Hangi sürümü kullandığın `SURUM.txt` dosyasında yazar.

**Windows:** proje klasöründe `kurulum.bat` dosyasına çift tıkla. Ya da terminalde:

```bash
py -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Kurulumu doğrula (yaklaşık 20 saniye, hepsi `passed` olmalı):

```bash
.venv\Scripts\python -m pytest -q
```

Proje kendi `.venv` klasörünü kullanır; bilgisayardaki diğer Python projelerine dokunmaz.

## 2. Uygulamayı açma

**Windows:** `baslat.bat` dosyasına çift tıkla. Ya da:

```bash
.venv\Scripts\python app.py
```

macOS / Linux: `.venv/bin/python app.py`

| Sekme | Ne yapılır |
|---|---|
| **Ayarlar** | Ajanı seç, etiket ver (ör. `ana`, `adim-cezasi-0.05`), bölüm sayısı, tekrar sayısı (varsayılan 3), ödüller, hiperparametreler |
| **Eğitim** | Eğit / Durdur. Sağda canlı öğrenme eğrisi; eğitim sırasında kaydedilen bölümleri istediğin hızda izle (hız yalnız izlemeyi etkiler, eğitimi değil) |
| **Test** | Eğitilmiş bir çalışmayı 100 normal (8x8) ve 100 büyük (12x12) labirentte random ve sağ el ile karşılaştırır. Bitince bildirim gelir. İstediğin test bölümünü izleyebilirsin |
| **Sonuçlar** | Bütün çalışmalar, grafikler, zip dışa/içe aktarma, rapor dışa aktarma, silme (Geri Dönüşüm Kutusu'na) |
| **Oyna** | Labirenti klavyeyle (ok tuşları / WASD) ajanın gördüğü kadarıyla oyna; problemi hissetmek için iyi |

**Dil:** sağ üstteki **English / Türkçe** düğmesi arayüzü ve bütün çıktıları (grafikler, tablolar, rapor) değiştirir. Seçim hatırlanır. Eğitim ya da test sürerken dil değiştirilemez; dil değişince Ayarlar sekmesindeki girilmiş değerler varsayılana döner.

Test labirentleri herkesin bilgisayarında aynıdır (sabit seed'ler, [configs/ortak_ayarlar.json](configs/ortak_ayarlar.json)); bu yüzden sonuçlar doğrudan karşılaştırılabilir.

## 3. Kendi ajanını yazma

Kısaca:
1. [agents/ornek_ajan.py](agents/ornek_ajan.py) dosyasını `agents/<algoritman>.py` olarak kopyala.
2. Sınıf adını ve `name` alanını değiştir (benzersiz olsun, ör. `"sarsa"`). Gözlem türünü `obs_mode` ile seç:
   - `"simple"`: tek tam sayı (0..11519), Q-tablosu gibi tablo yöntemleri için.
   - `"full"`: 0/1 vektör, sinir ağlı yöntemler için.
3. `act`, `learn`, `end_episode` içindeki TODO'ları doldur.
4. Uygulamayı aç (ya da Ayarlar'da **Yenile**): ajanın listede görünür.

`ornek` ajanı yalnız bir şablondur: rastgele hareket eder ve **hiçbir şey öğrenmez**. Onu eğitince sonucun iyileşmemesi normaldir.

Uygulamayı açmadan hızlı deneme (eğitir, sonra random ve sağ el ile karşılaştırır):

```bash
.venv\Scripts\python tools\ajan_dene.py --agent <name> --episodes 2000
```

Metodlar, gözlemin anlamı ve kurallar: [agents/README.md](agents/README.md).

## 4. Deney akışı

Herkesin izleyeceği adımlar ve sonuç tabloları: [DENEY_AKISI.md](DENEY_AKISI.md). Kısaca:

1. **Ana eğitim:** rastgele labirentlerde, ortak ayarlarla (etiket: `ana`).
2. **Test:** 100 normal + 100 büyük labirent, random ve sağ el ile karşılaştırma.
3. **Varyasyonlar:** sıfırdan **en az 2 kez** daha eğit, her seferinde **tek bir şeyi** değiştir (ör. adım cezası, duvar cezası, çıkışa yaklaşma ödülü `closer`). Etikete neyi değiştirdiğini yaz.
4. Varyasyonları aynı normal ve büyük labirentlerde test et.
5. En iyi ajanlar demoda hiç görmediği bir labirentte oynatılır.

## 5. Sonuçları gönderme

Raporu birleştiren kişiye her çalışma için iki şey gönder:

1. **Sonuç zip'i:** Sonuçlar sekmesinde çalışmayı seç → **Zip dışa aktar**. Zip; ayarları, eğitim günlüğünü, test sonuçlarını, kayıtlı bölümleri ve eğitilmiş modeli içerir.
2. **Ajan dosyan:** `agents/<algoritman>.py`. Alan kişi bu dosyayı kendi `agents/` klasörüne koymazsa grafikleri görebilir ama modelini test edemez ve oynatamaz.

Alan kişi: Sonuçlar → **İçe aktar** → zip ya da klasör seç.

**Rapor:** rapor kendiliğinden oluşmaz. Sonuçlar'da rapora girecek çalışmaları işaretle → **Rapor dışa aktar** → kaydedilecek yeri seç. Seçtiğin yerin içinde tek bir klasör açılır (`rapor_<tarih>`, İngilizcede `report_<tarih>`); içinde `rapor.html` (tarayıcıda kendiliğinden açılır), `rapor.md`, `grafikler/` ve `tablolar/` bulunur. Test edilmemiş çalışmalarda test tabloları ve baseline grafikleri boş kalır, önce **Test** sekmesinden test et.

Sonuçlar kendi bilgisayarında `results/` klasöründe durur.

### Yeni sürüm gelince

Ortamda düzeltme olursa yeni bir zip gelir. Eskisinin üstüne yazma:
1. Yeni zip'i **ayrı bir klasöre** çıkar.
2. Eski klasörden yenisine kendi dosyalarını taşı: `agents/<algoritman>.py` ve `results/` klasöründeki çalışmaların.
3. Yeni klasörde `kurulum.bat`'ı çalıştır (paketler değişmediyse hızlı biter).
4. Yeni sürümün `SURUM.txt`'sinde ya da gelen mesajda **test sonuçlarını etkileyen bir değişiklik** yazıyorsa, o sürümle tekrar test et; farklı sürümlerin sonuçlarını karşılaştırma.

## 6. Kurallar

- **Ortamı değiştirme** (`maze_env/`, `configs/ortak_ayarlar.json`). Özellikle test seed'leri değişirse herkesin sonucu karşılaştırılamaz hale gelir. Ortamda hata ya da eksik görürsen ortam sorumlusuna söyle.
- Ajan **yalnız gözlemi** kullanır; ortama (`env`) ve `info` içindeki `shortest_path` gibi alanlara bakarak karar vermez.
- **Hazır RL kütüphanesi yasak** (Stable-Baselines vb.). NumPy ve PyTorch serbest. PyTorch kullanacaksan kendin kur: `.venv\Scripts\python -m pip install torch`.
- Rastgelelik için `self.rng` kullan; yoksa sonuçlar tekrar üretilemez.
- Ödülleri Ayarlar sekmesinden değiştir, kodda değil: değerler sonuç klasörüne yazılır ve rapora girer.

## 7. Sorun giderme

| Belirti | Çözüm |
|---|---|
| `py` ya da `python` bulunamadı | Python'u python.org'dan kur, kurulumda "Add python.exe to PATH" kutusunu işaretle |
| Ajanım listede yok | Dosya `agents/` içinde mi, sınıf `Agent`'tan türüyor mu, `name` benzersiz mi? Ayarlar'da **Yenile**'ye bas; hata varsa orada yazar |
| `ModuleNotFoundError: PySide6` | Kurulum adımları atlanmış ya da `.venv` dışındaki Python ile açılmış; `kurulum.bat`'ı çalıştır, uygulamayı `baslat.bat` ile aç |
| Eğitim çok yavaş | Bölüm sayısını ya da tekrar sayısını deneme için düşür; izleme hızı eğitimi yavaşlatmaz |
| Testler kırılıyor | Ortam dosyalarını (`maze_env/`, `engine/`, `configs/`) değiştirdin mi? Zip'i yeniden çıkarıp dene |

## Klasörler

```text
app.py        uygulama
maze_env/     ortam (SPEC.md: kesin tanım)
agents/       ajanlar: herkesin kendi dosyası + şablon + kılavuz
baselines/    random, sağ el takibi
engine/       eğitim, test, kayıt, grafik, rapor
gui/          arayüz
configs/      ortak ayarlar (test labirentleri)
tools/        ajan_dene.py (komut satırı deneme), oyna_terminal.py
tests/        otomatik testler
results/      senin sonuçların
```
