"""Render the accompanying tables from the actual full-run CSV outputs."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def load(name):
    with (ROOT/"results"/(name+".csv")).open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))
def table(header,rows):
    return "\n".join("| "+" | ".join(row)+" |" for row in [header,["---"]*len(header),*rows])
def ci(r):
    def f(x):
        v=float(x)
        return f"{v:+.1f}" if abs(v)>=.05 else "0.0"
    return f'{f(r["mean"])}［{f(r["lo"])}, {f(r["hi"])}］'
def main():
    edge={(r["batch"],r["method"],r["metric"]):r for r in load("edge_assignment_summary")}
    trans={(r["batch"],r["method"],r["transition"],r["metric"]):r for r in load("transition_summary")}
    absolute={(r["batch"],r["method"],r["metric"]):r for r in load("absolute_summary_mm")}
    paired={(r["batch"],r["reference"],r["method"],r["metric"]):r for r in load("paired_contrasts")}
    labels={"no_change":"修正量为零","started_at_truth":"原本准确，修正后偏离","away_same_side":"停留在同侧，误差增大",
            "toward_no_cross":"接近或恰到真值，未越过","cross_gain":"越过真值，绝对误差减小",
            "cross_tie":"越过真值，绝对误差不变","cross_loss":"越过真值，绝对误差增大"}
    batches=["2014","2005-2006"]
    lines=["# 精确重配、边界贡献及零前向对照","## 四个落差 MAE（毫米）"]
    rows=[]
    for b in batches:
        for m in ("R","T"):
            fs=("baseline_mae_mm","constant_mae_mm","permuted_mae_mm","actual_mae_mm")
            rows.append([b,m]+[f'{float(edge[b,m,f]["mean"]):.2f}' for f in fs])
    lines.append(table(["批次","方法","全图 A0","加平均修正 A均","场内随机重配 A乱","实际对应 A实"],rows))
    lines.append("A乱 对每条边与同场所有修正量取平均，包含原来的修正量；它也是均匀随机排列的一阶期望。每场内部允许的配对数为边数的平方。为便于核对，主报告同时列出分解项与配对区间。[逐场四种误差及恒等式各项](results/edge_assignment_by_scene.csv)")
    lines.append("## 边界发生了什么")
    lines.append("以下各格依次为簇等权的边界比例、对全场净收益的贡献（mm）。每场的贡献分母为该场全部合格界面；逐场、逐簇求平均后仍可加回实测净收益。类别互斥，同分和零事件保留，表格取整。")
    rows=[]
    for c,label in labels.items():
        cells=[]
        for b,m in ((b,m) for b in batches for m in ("R","T")):
            frac=float(trans[b,m,c,"fraction"]["mean"])
            val=float(trans[b,m,c,"contribution_mm"]["mean"])
            fmt=lambda v:f"{v:.4f}" if 0<abs(v)<.005 else f"{v:.2f}"
            cells.append(f"{fmt(frac*100)}% ／ {fmt(val)}")
        rows.append([label]+cells)
    lines.append(table(["结果","2014 R","2014 T","2005/06 R","2005/06 T"],rows))
    lines.append("原本准确后偏离在 2014 出现 1 条。分类中的继续偏离和向真值移动均保留了误差由低估、高估出发的细分。[逐类贡献与区间](results/transition_summary.csv) · [按原本低估或高估拆分](results/sign_summary.csv) · [真实边界示例与坐标](results/boundary_examples.csv)")
    lines.append("## 只保留推远更新的空间代价")
    lines.append("UP 在全图逐像素取 max(B,T)。SHIFT 为 B 加全图正更新的算术均值。ROLL 把同一幅正更新图按图高和图宽各循环错位半幅再加回 B。三者在全图的平均加深量一致，边界子集内的均值可以不同。SHIFT 会给近侧和远侧都加同一个常数，理论上保持预测落差。")
    fs=["near","far","equal","jump","global"]
    header=["近侧","远侧","两侧等权","落差","全图"]
    for b in batches:
        lines.append("### "+b)
        rows=[]
        for m in ("B","UP","SHIFT","ROLL"):
            rows.append([m]+[f'{float(absolute[b,m,f+"_mae_mm"]["mean"]):.1f}' for f in fs])
        lines.append(table(["绝对 MAE（mm）"]+header,rows))
        rows=[]
        for ref,m in (("B","UP"),("B","SHIFT"),("B","ROLL"),("SHIFT","UP"),("ROLL","UP")):
            rows.append([ref+" 减 "+m]+[ci(paired[b,ref,m,f+"_mae_mm"]) for f in fs])
        lines.append(table(["配对收益［95% 区间］，mm"]+header,rows))
    lines.append("2014 年 UP 的两侧等权收益为 41.8 mm，统一加深取得 34.0 mm，UP 额外取得 7.8 mm［0.1, 16.3］。2005/06 年 UP 自身的全图 MAE 降低 8.0 mm［−8.7, 20.8］；统一加深降低 24.1 mm［15.6, 32.3］。边界两侧绝对深度、它们之间的落差与全图误差分别回答不同的问题。")
    (ROOT/"details.md").write_bytes(("\n\n".join(lines)+"\n").encode("utf8"))
if __name__=="__main__":main()
