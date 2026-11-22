# Deney Akışı (taslak)

Durum: **Konuşuldu, karar değil.** Proje (labirent) de henüz kesin değil.

## Genel kurallar
- 5 kişi, 5 farklı algoritma. Herkes kendi ajanını eğitir.
- Eğitim **rastgele labirentlerde** yapılır (her bölümde yeni labirent).
- Test labirentleri eğitimde hiç görülmemiş olmalı. Seed ile ayrılır, örnek:
  - Eğitim: seed 0-999
  - Normal boyut test: seed 10000-10099 (100 labirent)
  - Büyük boyut test: seed 20000-20099 (100 labirent)
- Herkes **aynı** test labirentlerini kullanır.
- Baseline'lar: **random** (her adımda rastgele yön) ve **sağ el takibi** (sağ eli hep duvarda). Bunlar eğitilmez.

## Her kişinin kendi ajanı için akış

1. **İlk eğitim:** Ajanı rastgele labirentlerde eğit.
2. **Normal boyut testi:** Ajanı 100 normal boyut test labirentinde koştur. Random ve sağ el de aynı 100 labirentte koşar. *(Madde 6: baseline karşılaştırması)*
3. **Büyük boyut testi:** Aynı ajanı, yeniden eğitmeden, 100 büyük boyut test labirentinde koştur. Random ve sağ el de aynı 100 labirentte koşar. *(Madde 8: genelleme)*
4. **1. tekrar eğitim:** Ajanı **sıfırdan** eğit, **sadece bir şeyi** değiştirerek (örnek: adım cezası yüksek). Gerisi ilk eğitimle aynı.
5. **1. tekrar eğitimin testi:** Bu ajanı 2. adımdaki **aynı** 100 normal labirentte ve 3. adımdaki **aynı** 100 büyük labirentte koştur. Sonuçları ilk ajanla karşılaştır. *(Madde 7: varyasyon)*
6. **2. tekrar eğitim:** Ajanı yine **sıfırdan** eğit, başka **tek bir şeyi** değiştirerek (örnek: duvara çarpma cezası yüksek).
7. **2. tekrar eğitimin testi:** 5. adımdaki gibi, aynı normal ve aynı büyük labirentlerde koştur, karşılaştır.

**En az 2 tekrar eğitim zorunlu** (ödev en az 2 varyasyon istiyor). Sonunda her kişinin elinde en az 3 ajan olur: ilk ajan + 2 varyasyon.

Notlar:
- Tekrar eğitim her zaman **sıfırdan** yapılır, eski ajanın üstüne değil. Yoksa farkın nereden geldiği anlaşılmaz.
- Her tekrar eğitimde **sadece bir şey** değişir. İki şey birden değişirse hangisinin etkisi olduğu bilinmez.
- Random ve sağ el öğrenmediği için sonuçları değişmez. 2. ve 3. adımdaki sayıları sonraki karşılaştırmalarda tekrar kullanabilirsin.

## Sonuç tabloları (her kişi için örnek)

**Normal boyut (100 labirent)**

| Yöntem | Çıkış oranı | Ortalama adım | Ortalama duvara çarpma |
|---|---|---|---|
| Random | | | |
| Sağ el takibi | | | |
| İlk ajan | | | |
| 1. varyasyon (adım cezası yüksek) | | | |
| 2. varyasyon (duvar cezası yüksek) | | | |

**Büyük boyut (100 labirent)**: aynı tablo.

## Final
5 kişi akışı bitirdikten sonra en iyi sonucu veren ajan, **hiç görmediği** rastgele bir labirentte hocaya canlı oynatılır. *(Madde 10: demo)* İstenirse aynı labirentte yanına sağ el takibi de oynatılıp karşılaştırma canlı gösterilir.
