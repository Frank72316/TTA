"""Render the saved null distribution; no resampling takes place here."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT=Path(__file__).resolve().parent

def read(name):
    with (ROOT/name).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def value(r,key):
    return float(r[key])

def display(r,key):
    return f"{100*value(r,key):.2f}%" if r["metric"]=="R" else f"{value(r,key):.2f}".rstrip("0").rstrip(".")

def graph(hist,summary):
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Noto Sans CJK SC","Microsoft YaHei","Noto Sans CJK JP","PingFang SC") if f in available),"DejaVu Sans")
    plt.rcParams.update({"font.family":[font,"DejaVu Sans"],"axes.spines.right":False,"axes.spines.top":False,
                         "axes.unicode_minus":False,"axes.edgecolor":"#77848a","font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),layout="constrained")
    for ax,metric,title in zip(axes,("R","same_nonzero"),("共同分量占交互平方和的比例","两张图同向且非零的图对数")):
        v=[r for r in hist if (r["scope"],r["design"],r["metric"])==("all","derangement",metric)]
        x=np.array([value(row,"value") for row in v])
        freq=np.array([int(row["frequency"]) for row in v])
        freq=freq/freq.sum()
        r=summary["all","derangement",metric]
        obs=value(r,"observed")
        mean=value(r,"exact_null_mean")
        if metric=="R":
            x*=100;obs*=100;mean*=100
            ax.hist(x,bins=np.arange(20.,92.),weights=freq,color="#228583",edgecolor="white",linewidth=.28)
            ax.set_xlabel("共同分量占比（%）")
            ax.set_xlim(27,89)
        else:
            ax.bar(x,freq,color="#228583",width=.85)
            ax.set_xlabel("同向图对数（共 64 对）")
            ax.set_xlim(13,45)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_ylabel("随机重配频率")
        ax.set_title(title,fontsize=12)
        ax.axvline(mean,color="#234857",lw=1.4,ls="--")
        ax.axvline(obs,color="#b64d21",lw=2)
        ax.text(obs-.01*(ax.get_xlim()[1]-ax.get_xlim()[0]),.86,"真实 "+display(r,"observed"),ha="right",
                color="#a2441b",transform=ax.get_xaxis_transform())
        ax.text(mean+.016*(ax.get_xlim()[1]-ax.get_xlim()[0]),.99,"随机均值 "+display(r,"exact_null_mean"),
                va="top",fontsize=9,transform=ax.get_xaxis_transform(),color="#234857")
    fig.suptitle("真实配对与 100 万次跨提示词随机重配",fontsize=16,fontweight="bold")
    fig.savefig(ROOT/"random_pairing.png",dpi=180)
    plt.close(fig)


def main():
    rows=read("summary.csv")
    sums={(r["scope"],r["design"],r["metric"]):r for r in rows}
    hist=read("null_distribution.csv")
    graph(hist,sums)
    real=sums["all","derangement","R"]
    sign=sums["all","derangement","same_nonzero"]
    clean=sums["no_conflict","derangement","R"]
    perm=sums["all","unrestricted","R"]
    m=read("mean_decomposition.csv")[0]
    lines=["# 夜景条件下，两张候选图是否有额外的同步响应？",
           f"同一原提示词下的两张候选图，在这 64 对样本中表现出额外同步性。保留全部单图响应、将 A 与其他提示词下的 B 重新匹配，共同分量占比的期望为 **{display(real,'exact_null_mean')}**；真实配对为 **{display(real,'observed')}**，高出 **{100*value(real,'excess_over_exact_mean'):.2f} 个百分点**。100 万次重配中，达到或超过真实值的次数为 {real['n_ge']}。",
           f"按图对计数也能看到差异：真实配对有 **{display(sign,'observed')}/64 对同向且非零**，随机期望 **{display(sign,'exact_null_mean')} 对**。这里每对各投一票，响应幅度最大的图对与其余图对权重相同。",
           "![真实配对与跨提示词重配的分布](random_pairing.png)",
           "## 跨提示词对照",
           "单图交互 h 是追加夜景要求后，调亮 10% 所引起的评分变化又改变了多少。对每次重配，64 个 A 固定，B 完整地一对一重排。每张图保留原来在自己提示词下测得的 h，包括符号、幅度和精确的零。本轮只重新匹配这些 h。64 个原提示词、Q0 及其 token 输入均逐对唯一，限制 B 不回原位即保证来自其他提示词。",
           "| 样本 | 指标 | 真实配对 | 随机期望 | 随机分布中央 95% | 达到或超过真实值 / 100 万次 |",
           "|---|---|---:|---:|---:|---:|"]
    for scope,label in (("all","全部 64 对"),("no_conflict","无语义冲突 58 对")):
        for metric,name in (("R","共同分量占比"),("same_nonzero","同向非零对数")):
            r=sums[scope,"derangement",metric]
            lines.append(f"| {label} | {name} | {display(r,'observed')} | {display(r,'exact_null_mean')} | {display(r,'null_q025')}–{display(r,'null_q975')} | {r['n_ge']} |")
    lines.extend(["","区间是固定这批图、仅改变配对得到的参考范围。主对照严格排除原配对，尾部次数给出描述性的重配参照。作为补充，在 B 身份可交换的零假设下，允许保留原位的普通置换检验也有 "+perm["n_ge"]+" 次共同分量占比达到或超过实测值；单侧 Monte Carlo p = (命中数 + 1)/(100 万 + 1) ≈ 0.000001。有限模拟的零次命中不等于真实尾概率为零。",
                  "","## 83.24% 是怎么组成的？","",
                  "设 a、b 为两侧的单图响应，S = Σ(a²+b²)。所有重配的 S 相同，因此 R = 1/2 + Σ(ab)/S。把乘积和按两侧均值展开，真实配对具有如下算术分解。",
                  "",f"| 数学基准 | 两侧均值同号的贡献 | 去掉两侧均值后的配对贡献 | 实测 R |",
                  "|---:|---:|---:|---:|",
                  f"| 50.00% | +{100*value(m,'marginal_dot_contribution'):.2f} 个百分点 | +{100*value(m,'centered_pair_contribution'):.2f} 个百分点 | {display(real,'observed')} |","",
                  f"50% 来自共同与差异分量的定义。两侧的 h 均值分别为 {value(m,'mean_h_A'):.8f}、{value(m,'mean_h_B'):.8f}。可任意重配时，平均乘积就是两侧均值的乘积，故精确基线为 {display(perm,'exact_null_mean')}。严格排除原配对会移走一批偏大的对角线乘积，基线随之降到 {display(real,'exact_null_mean')}。前表的 29.64 个百分点使用严格跨提示词基线，本表的 29.18 个百分点使用不限原位的基线。两侧响应去均值后的 Pearson 相关为 {value(m,'pearson_r'):.3f}。这些量衡量响应间的关联，不能解释成某种机制的因果份额。",
                  "","## 可以怎样讲给队友","",
                  "“在这批图里，夜景要求改变了 HPS 对调亮的反应。同一提示词下的两张候选图，比随机跨提示词配成一对的图更容易同向变化。这个差距在忽略响应大小、只数方向时也能看到。”",
                  "","适用范围是 ImageRewardDB 的这 64 对图、HPS v2.1、冻结的 Q0/Q1 和 1.1 倍亮度。真实候选同时共享提示词，并可能共享题材、构图或其他生成特征。重配对照保留了整批图片偏负的趋势，但不能将额外同步性单独归因于文本或画面。重配后的图像没有跨提示词的排序或人工偏好含义。"])
    # Keep the Markdown table contiguous while separating prose with blank lines.
    start=next(i for i,line in enumerate(lines) if line.startswith("| 样本 |"))
    body="\n\n".join(lines[:start])+"\n\n"+"\n".join(lines[start:])+"\n"
    (ROOT/"report.md").write_text(body,encoding="utf8")

if __name__=="__main__":
    main()

