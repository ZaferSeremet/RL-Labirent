# RL Projesi - Konuşulan Kararlar (2026-11-21)

Durum: **KONUŞULDU, plan henüz yapılmadı.** Plan Opus ile yapılacak. Bu dosya o plan için girdi.

Kaynak ödev: `Reinforcement Learning Course Assignment.pdf` (bu klasörde)

## Ödevin özeti
- Grup: PDF 4 kişi diyor. **Hoca 5 kişiye izin verdi**, şartı: her üye farklı bir algoritma deneyecek.
- Simülasyon zorunlu, donanım isteğe bağlı.
- Ortamı (dinamikler, geçişler, bitiş koşulları) kendimiz yazmalıyız. Hazır ortam indirip hazır ajan eğitmek yetmez.
- Çekirdek RL algoritmasını kendimiz yazmalıyız. Hazır eğitim hattı (ör. Stable-Baselines3) yerine geçemez. NumPy/PyTorch serbest.
- Zorunlu: durum/eylem tasarımı, ödül tasarımı + istenmeyen teşvik analizi, öğrenme eğrileri, baseline karşılaştırması, en az 2 hiperparametre/ödül varyasyonu, genelleme/dayanıklılık testi, tekrar üretilebilirlik, demo ve herkesin bireysel sözlü savunması.
- Teslim: kod, teknik rapor, eğitim sonuçları, baseline karşılaştırması, deney analizi, demo, sözlü savunma.
- Herkes tüm projeyi (ortam, algoritma, ödül, sonuçlar, sınırlılıklar) anlatabilmeli.

## Seçilen proje
**Labirent navigasyonu (kısmi gözlemli), Proje 1 + Proje 11 karışımı.**
Eleneen alternatifler: stok yönetimi (yedek plan), trafik ışığı.

## Ortam kararları
- Her `reset()`'te rastgele labirent (DFS/recursive backtracker ile). Ortam Python, Gymnasium `Env` arayüzü.
- Çoklu hedef: 3 ödül + 1 çıkış, konumları her bölümde rastgele.
- **Ajan ödüllerin ve çıkışın yerini biliyor. Duvarları sadece çevresinde (5x5) görüyor** (kısmi gözlem).
- Zaman baskısı: katı adım limiti `T` + küçük adım maliyeti. (Adım maliyeti ile zaman baskısı büyük ölçüde aynı baskı, ikisini ayrıştırmak ya da sade tutmak bizim kararımız.)
- Eylem: 4 yön, ayrık.
- Gözlem: 5x5 yerel duvar haritası, ajan koordinatı, çıkış ve ödül koordinatları (toplananlar işaretli), kalan süre oranı.

### Ödül taslağı (ayarlanacak)
| Olay | Ödül |
|---|---|
| Her adım | -0.01 |
| Duvara çarpma | -0.05 |
| Ödül topla | +1 (opsiyonel: zamanla azalan) |
| Çıkışa ulaş | +5, bölüm biter |
| Süre bitti, çıkışa ulaşılmadı | -2 |

### Parametreler (deney ve genelleme için)
`maze_size`, `num_rewards`, `view_radius`, `max_steps`, `mine_density` (varsayılan 0).

- **Mayın:** opsiyonel eklenti. İlk sürümde kapalı, ortam ve algoritmalar çalışınca zorluk/genelleme deneyi olarak açılır.
- Ödüllerin/çıkışın konumunu gizleme seçenekleri (`reveal_exit`, `reveal_rewards`) zor mod olarak düşünüldü, ana ayar değil.
- `view_radius=0/1` (neredeyse kör) en zor mod olarak değerlendirildi, ana ayar değil.

### Beklenen istenmeyen teşvikler (ödül analizi için)
- Çıkışa gitmeyip süre bitene kadar ödül toplamak (çıkış bonusu düşükse).
- Çarpma cezası çok büyükse yerinde durmak.
- Adım maliyeti çok büyükse ödülleri hiç toplamamak.

## Algoritmalar (5 kişi, herkes farklı)
Öneri dağılım (kimin hangisini alacağı henüz belli değil):
1. Tablo Q-learning (en kolay)
2. SARSA (en kolay)
3. DQN (orta)
4. REINFORCE (orta)
5. PPO veya A2C (en zor, en deneyimli kişiye)

Notlar:
- Tablo yöntemleri büyük durum uzayında öğrenemez. Küçük labirent (6x6), sade gözlem (4 komşu + hedefe göreli yön) gerekebilir. Gözlem farklıysa karşılaştırma adil olmaz, raporda belirtilmeli.
- Adil karşılaştırma için sabit: ortam, ödül, eğitim adım bütçesi, test labirentleri (sabit seed). Her algoritma 3-5 seed ile koşulmalı.
- Öğrenemeyen algoritma da geçerli sonuç, "neden öğrenemedi" analizi raporlanır.

## Baseline
Rastgele yürüyüş, duvar takibi, bilinen konumlara greedy giden basit sezgisel.

## Önerilen çalışma sırası
1. Önce kolay ortam (mayınsız, tam gözlem), her algoritma çalışsın.
2. Sonra kısmi gözlem ve çoklu hedef, sonra zorluk parametreleri.
3. Ortak `MazeEnv` + ortak değerlendirme scripti, herkes kendi algoritmasını aynı arayüze bağlar.

## Açık sorular (plan aşamasında netleşecek)
- Kim hangi algoritmayı alıyor?
- Labirent boyutu, `max_steps`, adım bütçesi için net sayılar.
- Ödül değerlerinin son hali.
- Demo formatı (matplotlib mı pygame mi), görev ve rapor bölüşümü.
- Teslim ve savunma tarihleri.
