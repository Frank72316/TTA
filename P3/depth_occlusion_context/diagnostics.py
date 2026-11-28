"""Generate a compact Chinese appendix directly from published tables, with no pixel cache."""
import csv
from pathlib import Path
root=Path(__file__).resolve().parent
def load(name):
    with (root/(name+".csv")).open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))
def table(header,rows):
    return "\n".join("| "+" | ".join(map(str,row))+" |" for row in [header,["---"]*len(header),*rows])
def m(r):
    return f'{float(r["mean"]):.1f}'
def ci(r):
    return f'{float(r["mean"]):+.1f}［{float(r["lo"]):+.1f}, {float(r["hi"]):+.1f}］'
def main():
    s={(r["batch"],r["method"],r["metric"]):r for r in load("summary")}
    batches=("2014","2005+2006","2005","2006")
    lines=["# 逐年对照与机制核查",
        "所有距离及误差以毫米计。正的配对收益表示后一方法误差更小。有符号残差按预测减真值，负数为低估。旧场景和复验场景分别统计；按具体布景系列整簇 bootstrap。"]
    for group in batches:
        lines.append("## "+group)
        rows=[]
        for method in ("B","T","T25","C","P","R","P_fixed","R_fixed"):
            fields=("all_near_mae","all_far_mae","all_mae","all_near_signed","all_far_signed","all_jump_mae","all_jump_signed","global_mae")
            rows.append([method]+[m(s[group,method,k]) for k in fields])
        lines.append(table(["方法","近侧 MAE","远侧 MAE","等权 MAE","近侧残差","远侧残差","跳变 MAE","跳变残差","全图 MAE"],rows))
        rows=[]
        for a,b in (("B","T"),("T","R"),("B","R"),("T","P"),("P","R"),("P_fixed","R_fixed"),("T25","R")):
            rows.append([a+" 减 "+b]+[ci(s[group,"paired",a+"_minus_"+b+"_"+k+"_mm"]) for k in ("all_near_mae","all_far_mae","all_mae")])
        lines.append(table(["配对收益［95% 区间］","近侧","远侧","等权"],rows))
        lines.append("事后以真值角色指定近侧读 B、远侧读 T，等权 MAE 为 "+m(s[group,"paired","posthoc_nearB_farT_mae_mm"])+" mm。它只用于描述角色差异，部署时没有真值角色标签。")
    lines+=["## 旧四窗 T 相对全图 B 如何移动深度",
        "每张图的两种预测取自完全相同的一组近侧或远侧像素。下表先在图内算移动比例、再在簇内和簇间平均。误差增量是对应移动方向像素的 MAE(T)−MAE(B)，正值为恶化；分组为空的图不进入该条件误差均值。改善、恶化和持平的像素数直接加总，仅描述覆盖。[逐图分母](pixel_movements.csv)"]
    rows=[]
    for r in load("movement_summary"):
        if r["batch"] not in batches[:2]:continue
        rows.append([r["batch"],{"near":"近侧","far":"远侧"}[r["side"]],{"nearer":"推近","farther":"推远","unchanged":"不动"}[r["direction"]],
                     f'{100*float(r["frac_cluster"]):.5f}%',r["improved"],r["worsened"],r["tied"],
                     f'{float(r["mean_err_change_mm"]):+.1f}' if r["mean_err_change_mm"] else "—"])
    lines.append(table(["批次","角色","方向","比例","误差改善像素","恶化像素","持平像素","条件误差增量 mm"],rows))
    lines+=["## 时间、输入与覆盖",
        "部署实测从已解码 RGB 起，含预处理、前向、核心回映和融合；不含 GT、校验与缓存读写。2014 的 B、T 直接引用已签名旧计时，R/P 各加同一旧 B 耗时；新年份共享同图实测 B，R/P 的其余步骤在同一 GPU 成对计时。真实图 T25 为 T 路径替换最终融合开销的分项估计。另给出合成图各策略完整端到端独立实测，每项一遍预热、四遍测量。表中显存为峰值已分配显存。"]
    rows=[]
    for r in load("latency_summary"):
        if r["source"]=="real" and r["batch"] not in batches[:2]:continue
        rows.append([r["source"],r["batch"],r["method"],r["n"],*[f'{float(r[k]):.1f}' for k in ("median_ms","p90_ms","max_allocated_MiB")]])
    lines.append(table(["来源","样本/尺寸","方法","图/重复","中位 ms","P90 ms","峰值 MiB"],rows))
    lines.append("50 个场景的 200 个角窗全部通过 T/R/P 核心归一化张量精确相等检查，实际核心线性放大为 1.666–1.668 倍（约），R/P 覆盖原宽约 79.5%–80.5%、原高约 79.4%–79.5%。2005/06 的 27 张 view1，使用正确视差方向重投影的平均亮度误差逐张小于反向；各场邻接整数视差的深度量化步长中位为 3.3–6.7 mm。采点阈值和去重与 2014 相同；两库共 50/50 张有合格界面。完整逐年覆盖见 [覆盖表](coverage_summary.csv)。")
    (root/"diagnostics.md").write_bytes(("\n\n".join(lines)+"\n").encode())
if __name__=="__main__":main()
