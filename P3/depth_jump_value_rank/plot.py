"""Draw the pre-specified millimetre decomposition from paired cluster summaries."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def main():
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next(n for n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "Droid Sans Fallback") if n in fonts)
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
    with (ROOT/"results"/"summary.csv").open(encoding="utf8") as f:
        data = {(r["batch"], r["metric"]): r for r in csv.DictReader(f) if r["scale"] == "native_R"}
    metrics = [("D", "总优势 D"), ("V", "数值贡献 V"), ("S", "排序贡献 S"), ("mean_gain", "只换档均值")]
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.25), sharex=True, sharey=True)
    for ax, (batch, title) in zip(axes, [("2005-2006", "2005/06 · 17 个场景簇"), ("2014", "2014 · 23 个场景簇")]):
        ax.axvline(0, color="#525d65", linewidth=1)
        ax.axhline(2.55, color="#b7c3c6", linewidth=.8, linestyle="--")
        for y, (key, _) in enumerate(metrics):
            row = data[batch, key]
            p, lo, hi = (float(row[k]) for k in ("mean", "lo", "hi"))
            color = "#705585" if key == "mean_gain" else "#166a79"
            ax.errorbar(p, y, xerr=[[p-lo], [hi-p]], fmt="o", color=color, capsize=4, elinewidth=2, ms=6)
            ax.annotate(f"{p:+.2f}", (p, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", color=color, fontsize=9)
        ax.set(xlim=(-85,105), ylim=(3.55,-.52), xticks=[-75,-50,-25,0,25,50,75,100],
               yticks=range(len(metrics)), yticklabels=[s for _,s in metrics], title=title)
        ax.grid(axis="x", alpha=.12)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.supxlabel("MAE 降低量（mm）     正值有利于低通核心", fontsize=11)
    fig.subplots_adjust(left=.13, right=.98, top=.88, bottom=.21, wspace=.12)
    fig.savefig(ROOT/"decomposition.png", dpi=195, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
