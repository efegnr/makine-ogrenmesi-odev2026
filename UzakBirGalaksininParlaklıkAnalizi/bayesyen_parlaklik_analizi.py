"""
YZM212 Makine Öğrenmesi - 4. Laboratuvar Ödevi
Uzak Bir Galaksinin Parlaklık Analizi (Bayesyen Çıkarım + MCMC)

Bu script, ödev dökümanındaki senaryonun tam uygulamasını içerir:
  - Sentetik gözlem verisi üretimi
  - Log-Likelihood, Log-Prior, Log-Posterior tanımları
  - MCMC örnekleyici (Metropolis-Hastings, çoklu walker ile - emcee mantığına benzer)
  - Corner plot (parametreler arası eklem ve marjinal dağılımlar)
  - Tüm yan analizler (dar prior etkisi, n_obs=5 etkisi)
  - Rapor için sayısal sonuçların yazdırılması

Not: emcee ve corner paketleri bu ortamda yüklü olmadığı için aynı mantığı saf NumPy
ile uyguluyoruz. Çıktılar, emcee/corner ile alınacak olanlarla aynı istatistiksel
niteliğe sahiptir (posterior medyanı, %16/%84 yüzdelikleri, güven aralıkları vb.).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) Çıktı klasörü
# ---------------------------------------------------------------------------
OUT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Sentetik veri üretimi (ödev dökümanındaki değerlerle birebir)
# ---------------------------------------------------------------------------
true_mu = 150.0       # Gerçek parlaklık
true_sigma = 10.0     # Gözlem hatası (standart sapma)
n_obs = 50            # Gözlem sayısı

np.random.seed(42)
data = true_mu + true_sigma * np.random.randn(n_obs)

print("=" * 70)
print("SENTETİK VERİ ÖZETİ")
print("=" * 70)
print(f"Gerçek mu (parlaklık)  : {true_mu}")
print(f"Gerçek sigma (gürültü) : {true_sigma}")
print(f"Gözlem sayısı (n_obs)  : {n_obs}")
print(f"Veri ortalaması        : {data.mean():.3f}")
print(f"Veri std.sapması       : {data.std(ddof=1):.3f}")
print()


# ---------------------------------------------------------------------------
# 2) Bayesyen fonksiyonlar (ödev dökümanındaki tanımlar)
# ---------------------------------------------------------------------------
def log_likelihood(theta, data):
    """Gaussian likelihood: p(D|theta) -> log biçiminde."""
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))


def log_prior_geniş(theta):
    """Geniş (informative olmayan) prior: mu ∈ (0, 300), sigma ∈ (0, 50)."""
    mu, sigma = theta
    if 0 < mu < 300 and 0 < sigma < 50:
        return 0.0
    return -np.inf


def log_prior_dar(theta):
    """Dar prior: mu ∈ (100, 110), sigma ∈ (0, 50). (Soru 6.1 için)"""
    mu, sigma = theta
    if 100 < mu < 110 and 0 < sigma < 50:
        return 0.0
    return -np.inf


def log_probability(theta, data, log_prior_fn):
    lp = log_prior_fn(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)


# ---------------------------------------------------------------------------
# 3) MCMC: Çoklu-walker, Metropolis-Hastings örnekleyici
#    (emcee'nin EnsembleSampler'ına istatistiksel olarak denk çıktılar verir)
# ---------------------------------------------------------------------------
def run_mcmc(data, log_prior_fn, n_walkers=32, n_steps=2000, initial=(140.0, 5.0),
             proposal_scale=(1.5, 0.8), seed=123):
    """Basit ama sağlam bir çoklu-walker Metropolis-Hastings örnekleyici."""
    rng = np.random.default_rng(seed)
    ndim = 2

    # Başlangıç pozisyonları (ödevdeki initial + küçük jitter)
    pos = np.array(initial) + 1e-4 * rng.standard_normal((n_walkers, ndim))

    # Her walker için başlangıç log-posterior
    log_prob = np.array([log_probability(p, data, log_prior_fn) for p in pos])

    chain = np.zeros((n_steps, n_walkers, ndim))
    accept = np.zeros(n_walkers)

    prop_scale = np.array(proposal_scale)

    for step in range(n_steps):
        for w in range(n_walkers):
            # Gaussian proposal
            proposal = pos[w] + prop_scale * rng.standard_normal(ndim)
            lp_new = log_probability(proposal, data, log_prior_fn)
            log_alpha = lp_new - log_prob[w]
            if np.log(rng.random()) < log_alpha:
                pos[w] = proposal
                log_prob[w] = lp_new
                accept[w] += 1
        chain[step] = pos

    acc_rate = accept.mean() / n_steps
    return chain, acc_rate


def flatten_chain(chain, discard=500, thin=15):
    """Burn-in at, seyrelt, walker'ları birleştir -> (N, ndim) düz örnek kümesi."""
    trimmed = chain[discard::thin]  # (n_kept_steps, n_walkers, ndim)
    return trimmed.reshape(-1, trimmed.shape[-1])


# ---------------------------------------------------------------------------
# 4) Custom corner plot (corner paketi yüklü olmadığı için kendimiz çiziyoruz)
# ---------------------------------------------------------------------------
def corner_plot(samples, labels, truths=None, title=None, save_path=None):
    """Marjinal histogramlar + eklem dağılım (2x2 grid)."""
    mu_s = samples[:, 0]
    sg_s = samples[:, 1]

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08,
                          width_ratios=[1, 1], height_ratios=[1, 1])

    ax_x = fig.add_subplot(gs[0, 0])
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_y = fig.add_subplot(gs[1, 1])
    gs_empty = fig.add_subplot(gs[0, 1])
    gs_empty.axis("off")

    # Üstteki marjinal (mu)
    ax_x.hist(mu_s, bins=50, color="#4C72B0", alpha=0.85)
    ax_x.set_xticklabels([])
    ax_x.set_ylabel("Sıklık")
    q16, q50, q84 = np.percentile(mu_s, [16, 50, 84])
    for q in (q16, q50, q84):
        ax_x.axvline(q, color="k", ls="--", lw=1, alpha=0.7)
    if truths is not None:
        ax_x.axvline(truths[0], color="red", lw=1.6, label=f"Gerçek={truths[0]}")
        ax_x.legend(fontsize=8, loc="upper right")
    ax_x.set_title(f"{labels[0]}: {q50:.2f}  (+{q84-q50:.2f}/-{q50-q16:.2f})",
                   fontsize=10)

    # Sağdaki marjinal (sigma) — yatay histogram
    ax_y.hist(sg_s, bins=50, orientation="horizontal", color="#4C72B0", alpha=0.85)
    ax_y.set_yticklabels([])
    ax_y.set_xlabel("Sıklık")
    q16s, q50s, q84s = np.percentile(sg_s, [16, 50, 84])
    for q in (q16s, q50s, q84s):
        ax_y.axhline(q, color="k", ls="--", lw=1, alpha=0.7)
    if truths is not None:
        ax_y.axhline(truths[1], color="red", lw=1.6, label=f"Gerçek={truths[1]}")
        ax_y.legend(fontsize=8, loc="upper right")
    ax_y.set_title(f"{labels[1]}: {q50s:.2f}  (+{q84s-q50s:.2f}/-{q50s-q16s:.2f})",
                   fontsize=10)

    # Eklem (joint) dağılım — 2D hist + kontur
    ax_xy.hist2d(mu_s, sg_s, bins=60, cmap="Blues")
    # %68 ve %95 güven bölgeleri için yaklaşık konturlar
    H, xe, ye = np.histogram2d(mu_s, sg_s, bins=60)
    Hs = np.sort(H.ravel())[::-1]
    cdf = np.cumsum(Hs) / Hs.sum()
    try:
        lv68 = Hs[np.searchsorted(cdf, 0.68)]
        lv95 = Hs[np.searchsorted(cdf, 0.95)]
        ax_xy.contour(
            0.5 * (xe[:-1] + xe[1:]),
            0.5 * (ye[:-1] + ye[1:]),
            H.T, levels=sorted({lv95, lv68}), colors="k", linewidths=1.0,
        )
    except Exception:
        pass

    if truths is not None:
        ax_xy.axvline(truths[0], color="red", lw=1.2, alpha=0.8)
        ax_xy.axhline(truths[1], color="red", lw=1.2, alpha=0.8)
        ax_xy.plot(truths[0], truths[1], "rs", ms=6)

    ax_xy.set_xlabel(labels[0])
    ax_xy.set_ylabel(labels[1])

    # X eksen hizalaması
    ax_x.set_xlim(ax_xy.get_xlim())
    ax_y.set_ylim(ax_xy.get_ylim())

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5) Ana deney: Geniş prior, n_obs=50 (ödevin temel deneyi)
# ---------------------------------------------------------------------------
print("=" * 70)
print("ANA DENEY: Geniş Prior,  n_obs = 50")
print("=" * 70)

chain_main, acc_main = run_mcmc(
    data, log_prior_fn=log_prior_geniş,
    n_walkers=32, n_steps=2000,
    initial=(140.0, 5.0), proposal_scale=(1.5, 0.8), seed=123,
)
flat_main = flatten_chain(chain_main, discard=500, thin=15)

mu_s = flat_main[:, 0]
sg_s = flat_main[:, 1]

mu_q16, mu_med, mu_q84 = np.percentile(mu_s, [16, 50, 84])
sg_q16, sg_med, sg_q84 = np.percentile(sg_s, [16, 50, 84])

mu_q025, mu_q975 = np.percentile(mu_s, [2.5, 97.5])
sg_q025, sg_q975 = np.percentile(sg_s, [2.5, 97.5])

print(f"\nOrtalama kabul oranı: {acc_main:.3f}")
print(f"Toplam posterior örnek sayısı: {flat_main.shape[0]}")
print()
print(f"mu (Parlaklık)    -> Median = {mu_med:.3f}")
print(f"                     %68 güven aralığı [%16, %84] = [{mu_q16:.3f}, {mu_q84:.3f}]")
print(f"                     %95 güven aralığı [%2.5, %97.5] = [{mu_q025:.3f}, {mu_q975:.3f}]")
print(f"                     Belirsizlik (+/-) ≈ +{mu_q84-mu_med:.3f} / -{mu_med-mu_q16:.3f}")
print(f"                     Mutlak Hata = {abs(mu_med - true_mu):.3f}  "
      f"(Oransal = %{100*abs(mu_med-true_mu)/true_mu:.3f})")
print()
print(f"sigma (Gürültü)   -> Median = {sg_med:.3f}")
print(f"                     %68 güven aralığı [%16, %84] = [{sg_q16:.3f}, {sg_q84:.3f}]")
print(f"                     %95 güven aralığı [%2.5, %97.5] = [{sg_q025:.3f}, {sg_q975:.3f}]")
print(f"                     Belirsizlik (+/-) ≈ +{sg_q84-sg_med:.3f} / -{sg_med-sg_q16:.3f}")
print(f"                     Mutlak Hata = {abs(sg_med - true_sigma):.3f}  "
      f"(Oransal = %{100*abs(sg_med-true_sigma)/true_sigma:.3f})")
print()

# Korelasyon (Corner Plot yorumu için)
corr_mu_sigma = np.corrcoef(mu_s, sg_s)[0, 1]
print(f"Pearson korelasyonu (mu, sigma) : {corr_mu_sigma:+.4f}")
print("(|ρ| ≈ 0 -> bağımsız, elips dik; |ρ| büyük -> korelasyon, elips eğik)")

# ---------------------------------------------------------------------------
# 6) Görseller
# ---------------------------------------------------------------------------
# 6a) Gözlem verisi histogramı
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.hist(data, bins=15, color="#55A868", alpha=0.85, edgecolor="k")
ax1.axvline(true_mu, color="red", lw=2, label=f"Gerçek μ = {true_mu}")
ax1.axvline(data.mean(), color="blue", ls="--", lw=2,
            label=f"Örnek ortalaması = {data.mean():.2f}")
ax1.set_xlabel("Ölçülen Parlaklık")
ax1.set_ylabel("Frekans")
ax1.set_title(f"Sentetik Gözlem Verisi  (n={n_obs}, σ={true_sigma})")
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "01_veri_histogrami.png"), dpi=140)
plt.close(fig1)

# 6b) Trace plots (zincirlerin karışma tanılaması)
fig2, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for w in range(chain_main.shape[1]):
    axes[0].plot(chain_main[:, w, 0], color="#4C72B0", alpha=0.3, lw=0.6)
    axes[1].plot(chain_main[:, w, 1], color="#C44E52", alpha=0.3, lw=0.6)
axes[0].axhline(true_mu, color="k", ls="--", lw=1.2, label=f"Gerçek μ = {true_mu}")
axes[1].axhline(true_sigma, color="k", ls="--", lw=1.2, label=f"Gerçek σ = {true_sigma}")
axes[0].axvline(500, color="gray", ls=":", label="burn-in sınırı")
axes[1].axvline(500, color="gray", ls=":")
axes[0].set_ylabel("μ (Parlaklık)")
axes[1].set_ylabel("σ (Gürültü)")
axes[1].set_xlabel("MCMC adımı")
axes[0].legend(loc="upper right", fontsize=9)
axes[1].legend(loc="upper right", fontsize=9)
axes[0].set_title("MCMC Zincir İzleme Grafiği (Trace Plot) — 32 walker")
for a in axes:
    a.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "02_trace_plot.png"), dpi=140)
plt.close(fig2)

# 6c) ANA Corner plot
fig3 = corner_plot(
    flat_main,
    labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Hata)"],
    truths=[true_mu, true_sigma],
    title="Posterior Corner Plot — Geniş Prior, n=50",
    save_path=os.path.join(FIG_DIR, "03_corner_plot_ana.png"),
)
plt.close(fig3)

# ---------------------------------------------------------------------------
# 7) Analiz 1: DAR PRIOR etkisi (Soru 6.1)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALİZ 1: DAR PRIOR  (mu ∈ (100, 110))")
print("=" * 70)

chain_dar, _ = run_mcmc(
    data, log_prior_fn=log_prior_dar,
    n_walkers=32, n_steps=2000,
    initial=(105.0, 10.0), proposal_scale=(0.4, 0.8), seed=7,
)
flat_dar = flatten_chain(chain_dar, discard=500, thin=15)
mu_d_q16, mu_d_med, mu_d_q84 = np.percentile(flat_dar[:, 0], [16, 50, 84])
sg_d_q16, sg_d_med, sg_d_q84 = np.percentile(flat_dar[:, 1], [16, 50, 84])

print(f"mu (Parlaklık)  -> Median = {mu_d_med:.3f}  [{mu_d_q16:.3f}, {mu_d_q84:.3f}]")
print(f"sigma (Gürültü) -> Median = {sg_d_med:.3f}  [{sg_d_q16:.3f}, {sg_d_q84:.3f}]")
print("-> Dar ve yanlış prior, mu tahminini veriye rağmen kendi bölgesine hapseder;")
print("   sigma çok büyük görünür çünkü model kötü μ ile uyuşmaya gürültü ile direnir.")

# Karşılaştırma grafiği
fig4, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
axa.hist(flat_main[:, 0], bins=60, alpha=0.75, color="#4C72B0",
         label="Geniş prior [0, 300]")
axa.hist(flat_dar[:, 0], bins=60, alpha=0.75, color="#DD8452",
         label="Dar prior [100, 110]")
axa.axvline(true_mu, color="red", lw=2, label=f"Gerçek μ = {true_mu}")
axa.set_xlabel(r"$\mu$ (Parlaklık)")
axa.set_ylabel("Sıklık")
axa.set_title("Prior seçiminin posterior dağılıma etkisi — μ")
axa.legend()
axa.grid(alpha=0.3)

axb.hist(flat_main[:, 1], bins=60, alpha=0.75, color="#4C72B0",
         label="Geniş prior (mu)")
axb.hist(flat_dar[:, 1], bins=60, alpha=0.75, color="#DD8452",
         label="Dar prior (mu)")
axb.axvline(true_sigma, color="red", lw=2, label=f"Gerçek σ = {true_sigma}")
axb.set_xlabel(r"$\sigma$ (Gürültü)")
axb.set_ylabel("Sıklık")
axb.set_title("Prior seçiminin posterior dağılıma etkisi — σ")
axb.legend()
axb.grid(alpha=0.3)

fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "04_prior_etkisi.png"), dpi=140)
plt.close(fig4)

# ---------------------------------------------------------------------------
# 8) Analiz 2: n_obs = 5 etkisi (Soru 6.2)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALİZ 2: AZ VERİ  (n_obs = 5)")
print("=" * 70)

np.random.seed(42)
data_small = true_mu + true_sigma * np.random.randn(5)
chain_small, _ = run_mcmc(
    data_small, log_prior_fn=log_prior_geniş,
    n_walkers=32, n_steps=2000,
    initial=(140.0, 5.0), proposal_scale=(3.0, 2.5), seed=11,
)
flat_small = flatten_chain(chain_small, discard=500, thin=15)
mu_s5_q16, mu_s5_med, mu_s5_q84 = np.percentile(flat_small[:, 0], [16, 50, 84])
sg_s5_q16, sg_s5_med, sg_s5_q84 = np.percentile(flat_small[:, 1], [16, 50, 84])

width_mu_50 = mu_q84 - mu_q16
width_mu_5 = mu_s5_q84 - mu_s5_q16
width_sg_50 = sg_q84 - sg_q16
width_sg_5 = sg_s5_q84 - sg_s5_q16

print(f"n=5: mu    -> {mu_s5_med:.3f}  [{mu_s5_q16:.3f}, {mu_s5_q84:.3f}]  "
      f"(genişlik = {width_mu_5:.3f})")
print(f"n=5: sigma -> {sg_s5_med:.3f}  [{sg_s5_q16:.3f}, {sg_s5_q84:.3f}]  "
      f"(genişlik = {width_sg_5:.3f})")
print(f"n=50 karşılaştırma:")
print(f"n=50: mu    -> genişlik = {width_mu_50:.3f}  "
      f"(n=5 / n=50 oranı = {width_mu_5/width_mu_50:.2f}x)")
print(f"n=50: sigma -> genişlik = {width_sg_50:.3f}  "
      f"(n=5 / n=50 oranı = {width_sg_5/width_sg_50:.2f}x)")
print("-> n azaldıkça posterior belirsizlik BÜYÜR (1/sqrt(n) kuralıyla uyumlu).")

# Grafik: n=5 vs n=50 karşılaştırması
fig5, (axc, axd) = plt.subplots(1, 2, figsize=(12, 5))
axc.hist(flat_main[:, 0], bins=60, alpha=0.75, color="#4C72B0",
         label=f"n=50 (genişlik={width_mu_50:.2f})", density=True)
axc.hist(flat_small[:, 0], bins=60, alpha=0.65, color="#C44E52",
         label=f"n=5  (genişlik={width_mu_5:.2f})", density=True)
axc.axvline(true_mu, color="black", lw=2, label=f"Gerçek μ = {true_mu}")
axc.set_xlabel(r"$\mu$ (Parlaklık)")
axc.set_ylabel("Olasılık Yoğunluğu")
axc.set_title("Veri miktarının μ posteriorine etkisi")
axc.legend()
axc.grid(alpha=0.3)

axd.hist(flat_main[:, 1], bins=60, alpha=0.75, color="#4C72B0",
         label=f"n=50 (genişlik={width_sg_50:.2f})", density=True)
axd.hist(flat_small[:, 1], bins=60, alpha=0.65, color="#C44E52",
         label=f"n=5  (genişlik={width_sg_5:.2f})", density=True)
axd.axvline(true_sigma, color="black", lw=2, label=f"Gerçek σ = {true_sigma}")
axd.set_xlabel(r"$\sigma$ (Gürültü)")
axd.set_ylabel("Olasılık Yoğunluğu")
axd.set_title("Veri miktarının σ posteriorine etkisi")
axd.legend()
axd.grid(alpha=0.3)

fig5.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, "05_veri_miktari_etkisi.png"), dpi=140)
plt.close(fig5)

# ---------------------------------------------------------------------------
# 9) Rapor için tüm sonuçları JSON olarak kaydet
# ---------------------------------------------------------------------------
import json
results = {
    "ana_deney": {
        "true_mu": true_mu, "true_sigma": true_sigma, "n_obs": n_obs,
        "mu_median": float(mu_med), "mu_q16": float(mu_q16), "mu_q84": float(mu_q84),
        "mu_q025": float(mu_q025), "mu_q975": float(mu_q975),
        "mu_abs_err": float(abs(mu_med - true_mu)),
        "sigma_median": float(sg_med), "sigma_q16": float(sg_q16),
        "sigma_q84": float(sg_q84),
        "sigma_q025": float(sg_q025), "sigma_q975": float(sg_q975),
        "sigma_abs_err": float(abs(sg_med - true_sigma)),
        "corr_mu_sigma": float(corr_mu_sigma),
        "accept_rate": float(acc_main),
    },
    "dar_prior": {
        "mu_median": float(mu_d_med), "mu_q16": float(mu_d_q16),
        "mu_q84": float(mu_d_q84),
        "sigma_median": float(sg_d_med), "sigma_q16": float(sg_d_q16),
        "sigma_q84": float(sg_d_q84),
    },
    "n5": {
        "mu_median": float(mu_s5_med), "mu_q16": float(mu_s5_q16),
        "mu_q84": float(mu_s5_q84),
        "sigma_median": float(sg_s5_med), "sigma_q16": float(sg_s5_q16),
        "sigma_q84": float(sg_s5_q84),
        "width_mu_n50": float(width_mu_50), "width_mu_n5": float(width_mu_5),
        "width_sg_n50": float(width_sg_50), "width_sg_n5": float(width_sg_5),
    },
}
with open(os.path.join(OUT, "sonuclar.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("TÜM GRAFİKLER ve sonuclar.json KAYDEDİLDİ:")
print("=" * 70)
for fn in sorted(os.listdir(FIG_DIR)):
    print("  figures/" + fn)
print("  sonuclar.json")
