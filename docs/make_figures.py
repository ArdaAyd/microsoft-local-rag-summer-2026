"""README/EXPERIMENTS figürlerini üretir -> docs/assets/*.png

Sayılar deneylerden alınmıştır (kaynak yorumlarda). Yeni ölçüm yaparsan değerleri
güncelle ve `python docs/make_figures.py` ile yenile.

Tasarım: Okabe-Ito paleti (renk körlüğüne duyarlı), açık zemin (GitHub açık+koyu tema),
tek eksen, bar üstünde doğrudan değer etiketi. Başlık / alt başlık / lejant, grafiğin
ÜSTÜNDE ayrı yatay bantlara yerleştirilir -> barlarla ASLA çakışmaz.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

OI = {"orange": "#E69F00", "green": "#009E73", "blue": "#0072B2",
      "grey": "#C2C2C2", "ink": "#1a1a1a", "muted": "#6b6b6b"}

plt.rcParams.update({
    "font.size": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "#cccccc",
    "text.color": OI["ink"], "axes.labelcolor": OI["ink"],
    "xtick.color": OI["ink"], "ytick.color": OI["muted"],
})

# Grafik alanının dikey sınırları (figür koordinatı). Üstteki 0.72–1.0 bandı
# başlık/alt başlık/lejant için ayrılır; barlar bu banda hiç giremez.
PLOT_TOP = 0.72


def _axes(fig):
    ax = fig.add_axes([0.11, 0.14, 0.85, PLOT_TOP - 0.14])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ededed", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Recall / accuracy (%)")
    return ax


def _header(fig, title, subtitle):
    fig.text(0.035, 0.93, title, fontsize=15.5, fontweight="bold", color=OI["ink"])
    fig.text(0.035, 0.85, subtitle, fontsize=10.5, color=OI["muted"])


def _labels(ax, bars):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.6,
                f"{b.get_height():.0f}%", ha="center", va="bottom",
                fontsize=11.5, color=OI["ink"], fontweight="bold")


# ---- Figure 1: iyimserlik açığı (Deney 6 + 8, final sistem bge-m3) ----
def fig_optimism_gap():
    groups = ["Retrieval\n(kw@5)", "Answer\naccuracy"]
    original = [95.7, 92.9]
    realistic = [73.9, 70.0]
    x = range(len(groups)); w = 0.34

    fig = plt.figure(figsize=(7.4, 5.0))
    ax = _axes(fig)
    b1 = ax.bar([i - w / 2 for i in x], original, w, color=OI["orange"],
                label="Questions in the documents' words")
    b2 = ax.bar([i + w / 2 for i in x], realistic, w, color=OI["blue"],
                label="Realistic user phrasing")
    _labels(ax, b1); _labels(ax, b2)
    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    _header(fig, "The evaluation was optimistic",
            "Same system and targets — only the wording of the questions changed")
    # Lejant: başlık bandının altında, grafik alanının üstünde ayrı bir satır
    fig.legend(loc="center", bbox_to_anchor=(0.53, 0.775), ncol=2,
               frameon=False, fontsize=11, handlelength=1.2, columnspacing=2.4)
    fig.savefig(OUT / "optimism_gap.png", dpi=150)
    plt.close(fig)


# ---- Figure 2: embedding modeli karşılaştırması (Deney 7, karar metriği) ----
def fig_embedding_models():
    models = ["MiniLM-L12", "e5-small", "e5-base", "bge-m3"]
    kw5 = [60.9, 60.9, 65.2, 73.9]
    colors = [OI["grey"], OI["grey"], OI["grey"], OI["green"]]

    fig = plt.figure(figsize=(7.4, 5.0))
    ax = _axes(fig)
    bars = ax.bar(models, kw5, color=colors, width=0.6)
    _labels(ax, bars)
    _header(fig, "Only a stronger embedding model closed the gap",
            "Retrieval on realistic phrasing (paraphrased kw@5) — the decision metric")
    ax.annotate("adopted", xy=(3, 73.9), xytext=(3, 88), ha="center",
                fontsize=10.5, color=OI["green"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=OI["green"], lw=1.2))
    fig.savefig(OUT / "embedding_models.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_optimism_gap()
    fig_embedding_models()
    print("Yazıldı:", *(p.name for p in sorted(OUT.glob("*.png"))))
