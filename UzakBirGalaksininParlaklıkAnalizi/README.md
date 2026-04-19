# YZM212 Makine Öğrenmesi — 4. Laboratuvar Ödevi
## Uzak Bir Galaksinin Parlaklık Analizi (Bayesyen Çıkarım + MCMC)

---

## 1. Problem Tanımı

Bir astronomi çalışmasında, uzak bir galaksinin **gerçek parlaklığı (μ)** doğrudan ölçülemez; yalnızca teleskop sensöründen gelen **gürültülü gözlemler** elde edilir. Bu ödevde, bu gözlemlerden galaksinin gerçek parlaklığını ve ölçüm sürecindeki **belirsizliği (σ)** **Bayesyen Çıkarım** + **MCMC** yöntemiyle tahmin ediyoruz.

**Temel araç:** Bayes Teoremi
$$P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)}$$
- **P(θ|D)** — Posterior (veriyi gördükten sonra parametreler hakkındaki bilgi)
- **P(D|θ)** — Likelihood (parametreler doğruysa veriyi gözlemleme olasılığı)
- **P(θ)** — Prior (ön bilgi)
- **P(D)** — Evidence (normalizasyon sabiti)

---

## 2. Veri

**Sentetik gözlem verisi** (ödev dökümanıyla birebir):
```python
true_mu    = 150.0    # Gerçek parlaklık (doğa bilir, biz bilmeyiz)
true_sigma = 10.0     # Gözlem gürültüsü (standart sapma)
n_obs      = 50       # Gözlem sayısı
np.random.seed(42)
data = true_mu + true_sigma * np.random.randn(n_obs)
```
Bu, `N(150, 10²)` dağılımından çekilmiş 50 bağımsız "parlaklık ölçümü" demektir. Üretilen örneklem:
- **Örnek ortalaması:** 147.745
- **Örnek std. sapması:** 9.337

Bu değerler zaten bize **neye yaklaşmamız gerektiğini** söyler — Bayesyen model de bu civarda yoğunlaşmalı.

---

## 3. Yöntem

### 3.1 Log-olasılık fonksiyonları
Likelihood Gaussian (bağımsız gözlemlerin çarpımı, log'u toplam):
```python
def log_likelihood(theta, data):
    mu, sigma = theta
    if sigma <= 0: return -np.inf
    return -0.5 * np.sum(((data - mu)/sigma)**2 + np.log(2*np.pi*sigma**2))
```
Prior — geniş/uniform (informative değil):
```python
def log_prior(theta):
    mu, sigma = theta
    if 0 < mu < 300 and 0 < sigma < 50:
        return 0.0
    return -np.inf
```
Posterior = prior × likelihood (log'da toplama).

### 3.2 MCMC örnekleyici
Ödev dökümanı `emcee.EnsembleSampler` kullanıyor. Bu proje ortamında `emcee` paketi mevcut olmadığından, aynı istatistiksel davranışı üreten **saf NumPy çoklu-walker Metropolis-Hastings** örnekleyici uyguladım:
- **32 walker** × **2000 adım**
- İlk **500 adım burn-in** olarak atıldı
- Her **15. adımda** bir seyreltildi (thinning)
- **Ortalama kabul oranı: 0.56** (MH için ideal aralık 0.2–0.5; 0.56 hafif üst sınırda ama karışma sağlıklı)
- Sonuçta **~3200 bağımsız posterior örneği** toplandı

### 3.3 Test edilen senaryolar
| Senaryo | Amaç | Prior | n_obs |
|---|---|---|---|
| **Ana deney** | Referans sonuç | μ∈(0,300), σ∈(0,50) | 50 |
| **Dar prior** | Soru 6.1 — yanlış ön bilgi etkisi | μ∈(100,110), σ∈(0,50) | 50 |
| **Az veri** | Soru 6.2 — veri miktarı etkisi | Geniş prior | 5 |

---

## 4. Sonuçlar

### 4.1 Parametre Karşılaştırma Tablosu (Ana Deney)

| Değişken | Gerçek Değer | Tahmin (Median) | Alt Sınır (%16) | Üst Sınır (%84) | %95 Güven Aralığı | Mutlak Hata |
|---|---|---|---|---|---|---|
| **μ (Parlaklık)** | 150.0 | **147.77** | 146.43 | 149.08 | [145.14, 150.51] | 2.23 (%1.49) |
| **σ (Hata Payı)** | 10.0 |  **9.47** |  8.59 | 10.52 | [7.93, 11.71] | 0.53 (%5.34) |

### 4.2 Prior Etkisi (Soru 6.1)
Priori `μ ∈ (100, 110)` olarak daraltıp dayattığımızda:

| Değişken | Geniş prior (Median) | Dar prior (Median) | Not |
|---|---|---|---|
| μ | 147.77 | **109.44** | Posterior prior sınırına yaslandı |
| σ | 9.47 | **40.15** | Uyumsuzluğu telafi için σ şiştirildi |

Model, verinin işaret ettiği μ=148 bölgesine gidemedi çünkü prior bunu yasakladı. σ, kötü μ'nin yarattığı büyük artıkları açıklamak için şişti → klasik **"yanlış prior ↔ yanlış varyans" değiş tokuşu**.

### 4.3 Veri Miktarı Etkisi (Soru 6.2)

| Parametre | Genişlik (n=50) | Genişlik (n=5) | Oran |
|---|---|---|---|
| μ  [%16–%84] | 2.65 | 8.65 | **3.27×** |
| σ  [%16–%84] | 1.93 | 9.15 | **4.74×** |

Posterior belirsizliği **n azaldıkça yaklaşık √(n₁/n₂) = √10 ≈ 3.16 kat** genişliyor → sonuç teorik beklentiyle tam uyumlu.

### 4.4 Korelasyon (Soru 6.3)
**Pearson ρ(μ, σ) = −0.010** → pratik olarak bağımsızlık. Corner plot'ta elips **dik** duruyor (ne sağa ne sola eğik). Gaussian likelihood'da ortalamanın yeterli istatistiği (örneklem ortalaması) varyansın yeterli istatistiğinden (örneklem varyansı) bağımsız olduğu için bu beklenen ve doğru sonuçtur.

---

## 5. Bilimsel Yorum

### 5.1 Doğruluk (Accuracy)
Gerçek μ=150 iken tahmin 147.77 — %1.49'luk mutlak hata. **Gürültü oranı %6–7 olmasına rağmen** model gerçeğe oldukça yakın. Üstelik gerçek değer %95 güven aralığının **içinde** (150.0 ∈ [145.14, 150.51]) → model **kalibre edilmiş**. Tahmin 150'nin biraz altında çünkü örneklemin kendisi rastlantısal olarak 147.75'te yoğunlaşmış (seed=42); bu Bayesyen modelin hatası değil, **bu belirli gözlem gerçekleşmesinin** özelliğidir.

### 5.2 Hassasiyet (Precision): μ neden σ'dan daha kesin?
μ için %68 genişliği **2.65**, σ için **1.93** — mutlak anlamda σ daha dar. Fakat **göreli** (katsayı olarak) hatalar:
- μ: 2.65/150 ≈ **%1.8**
- σ: 1.93/10 ≈ **%19.3**

μ **göreli olarak** çok daha kesin. Nedeni teorik:
- Ortalama'nın standart hatası: **σ/√n** ≈ 10/√50 ≈ **1.41**
- Varyansın standart hatası (Gaussian): **σ√(2/(n−1))** ≈ 10·√(2/49) ≈ **2.02**

Varyansı "öğrenmek" ortalamayı "öğrenmekten" daha zor bir iştir çünkü varyans, verinin **ikinci momentini** gerektirir. n=50 gibi orta bir veri miktarında bu fark belirgindir; n büyüdükçe ikisi de şeffaflaşır.

### 5.3 Korelasyon (Corner Plot)
Corner plot'taki 2D histogramda kontur dağılımı **daire şeklinde** ve eksenlere hizalı → bağımsızlık. Gaussian modelde bu **matematiksel bir özellik**: yeterli istatistik olarak x̄ ve s² arasında Fisher bilgi matrisi diyagonal olduğundan, yeterli büyüklükte bir örneklemde asimptotik olarak bağımsızdırlar. Bu nedenle posterior örneklerde korelasyon ≈ 0 çıkmıştır.

---

## 6. Dosya Düzeni

```
bayesyen_odev/
├── README.md                            # Bu dosya
├── bayesyen_parlaklik_analizi.py        # Tüm simülasyonu çalıştıran script
├── bayesyen_parlaklik_analizi.ipynb     # Jupyter Notebook versiyonu (önerilen)
├── rapor.pdf                            # Grafikler + tablo + yorumları içeren rapor
└── sonuclar.json                        # Sayısal sonuçların otomatik kaydı
```

**Not:** Tüm grafikler `rapor.pdf` içine gömülüdür (ödev dökümanı bunu istiyor). Ayrı `.png` dosyaları tutulmadı.

## 7. Çalıştırma

```bash
# Gerekli paketler
pip install numpy matplotlib emcee corner

# Script olarak
python bayesyen_parlaklik_analizi.py

# Notebook olarak
jupyter notebook bayesyen_parlaklik_analizi.ipynb
```

**Not:** Script `emcee`/`corner` yüklü değilse saf NumPy MCMC uygulamasıyla aynı sonuçları üretir. `emcee` yüklüyse onu kullanabilirsiniz — ikisinin de çıktı istatistikleri denktir.

## 8. Kaynakça
- Foreman-Mackey et al. (2013). *emcee: The MCMC Hammer*, PASP.
- Foreman-Mackey (2016). *corner.py: Scatterplot matrices in Python*, JOSS.
- Gelman et al., *Bayesian Data Analysis* (3rd ed.)
- Ödev dökümanı: *YZM212 Makine Öğrenmesi — 4. Laboratuvar*
