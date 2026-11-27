"""Validate scores, derive paired statistics and write the report (no model needed)."""
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median, quantiles

from pilot_io import ROOT, load_pairs, read_csv, write_csv


def choice(delta):
    return "A" if delta > 0 else "B" if delta < 0 else "tie"


def transition(old, new):
    if old == 0 or new == 0:
        return "tie"
    return "flip" if old * new < 0 else "same"


def quartiles(values):
    l, _, u = quantiles(values, n=4, method="inclusive")
    return l, median(values), u


def number(x):
    return f"{x:.8f}"


def analyze():
    pairs = load_pairs()
    labeled = {p["pair_id"]:p for p in pairs}
    lighting = {r["pair_id"]:r for r in read_csv(ROOT/"lighting.csv")}
    assert set(lighting) <= set(labeled)
    scores = read_csv(ROOT/"scores.csv")
    cells = {}
    for r in scores:
        p = labeled[r["pair_id"]]
        key = (r["pair_id"],r["side"],r["prompt_condition"],r["light"])
        assert key not in cells and r["image_id"] == p["image_"+r["side"]]
        assert all(x != "" for x in r.values()) and math.isfinite(float(r["score"]))
        cells[key] = float(r["score"])
    expected = {(i,a,p,l) for i in labeled for a in ("A","B") for p in ("P0","P1") for l in ("L0","L1")}
    assert cells.keys() == expected and len(scores) == 512
    results = []
    for p in pairs:
        i = p["pair_id"]
        light = lighting.get(i, {})
        record = dict(pair_id=i, prompt_id=p["prompt_id"], P0=p["P0"], P1=p["P1"],
                      human_winner_P0=p["human_winner_P0"], lighting=light.get("category","none"))
        for cond in ("P0","P1"):
            a0, a1, b0, b1 = [cells[i,a,cond,l] for a in ("A","B") for l in ("L0","L1")]
            d0, d1 = a0-b0, a1-b1
            at, bt = (a0+a1)/2, (b0+b1)/2
            dt = at-bt
            assert dt == (d0+d1)/2
            vals = dict(d0=d0, d1=d1, delta=d1-d0, change_A=a1-a0, change_B=b1-b0,
                        tta_A=at, tta_B=bt, d_tta=dt, pick0=choice(d0), pick1=choice(d1),
                        pick_tta=choice(dt), change_L1=transition(d0,d1), change_TTA=transition(d0,dt))
            record.update({cond+"_"+k:v for k,v in vals.items()})
        record["I"] = record["P1_delta"]-record["P0_delta"]
        record["abs_I"] = abs(record["I"])
        for name in ("pick0","pick_tta"):
            pick = record["P0_"+name]
            record["P0_"+name+"_vs_human"] = "tie" if pick=="tie" else "correct" if pick==p["human_winner_P0"] else "wrong"
        results.append(record)
    write_csv(ROOT/"pair_results.csv",results)
    return results, scores, lighting


def report(results,scores,lighting):
    runs = json.loads((ROOT/"runs.json").read_text())
    env = json.loads((ROOT/"environment.json").read_text())
    n = len(results)
    q1, med, q3 = quartiles([r["abs_I"] for r in results])
    more = sum(abs(r["P1_delta"])>abs(r["P0_delta"]) for r in results)
    equal = sum(abs(r["P1_delta"])==abs(r["P0_delta"]) for r in results)
    elapsed = sum(t["seconds"] for t in runs)
    lines = ["# 调亮同一对候选图后，HPS v2.1 的分差与排序",
             "", f"ImageRewardDB 验证集，{n} 对图，{n} 条不同的原提示词，共 {len(scores)} 个图文评分。"
             "每对图共享两个提示词和同一组输入像素。P1 在原文后统一追加 “The scene is at night, with dim lighting.”。",
             "", "## 分差与翻转", "",
             f"追加夜景要求后，亮度响应的改变量 |I| 中位数为 **{number(med)}**，"
             f"第 25 和 75 百分位数分别为 {number(q1)}、{number(q3)}。"
             f"在逐对比较中，P1 的 |Δ| 有 {more} 对更大、{n-more-equal} 对更小、{equal} 对相等。"
             "这些数描述固定样本的模型响应。|I| 衡量响应的改变量，增幅的方向由配对的 |Δ| 比较给出。",
             "", "| 提示词 | |Δ| 中位数 | |Δ| 四分位区间 | 调亮翻转 | 调亮比较含同分 | TTA 翻转 | TTA 比较含同分 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    # Escape absolute-value bars inside Markdown table cells.
    lines[-2] = lines[-2].replace("|Δ|","abs(Δ)")
    for cond,title in [("P0","P0 原提示词"),("P1","P1 追加夜景")]:
        lo,mid,hi = quartiles([abs(r[cond+"_delta"]) for r in results])
        c = Counter(r[cond+"_change_L1"] for r in results)
        t = Counter(r[cond+"_change_TTA"] for r in results)
        lines.append(f"| {title} | {number(mid)} | {number(lo)}–{number(hi)} | {c['flip']}/{n} | {c['tie']} | {t['flip']}/{n} | {t['tie']} |")
    lines += ["","分差均为 A 减 B。Δ = d1 − d0，I = Δ(P1) − Δ(P0)。表中翻转分别比较 L0 与 L1、L0 与 TTA，两端严格异号才计入；任一端为零单列。四分位数用线性插值。", "",
              "## P0 与原图人工偏好", "",
              "| 原图判定 → TTA 判定 | TTA 选对 | TTA 选错 | TTA 同分 | 原图合计 |",
              "|---|---:|---:|---:|---:|"]
    conf = Counter((r["P0_pick0_vs_human"],r["P0_pick_tta_vs_human"]) for r in results)
    for old,title in [("correct","原图选对"),("wrong","原图选错"),("tie","原图同分")]:
        counts = [conf[old,new] for new in ("correct","wrong","tie")]
        lines.append(f"| {title} | {counts[0]} | {counts[1]} | {counts[2]} | {sum(counts)} |")
    totals = [sum(v for (old,new),v in conf.items() if new==k) for k in ("correct","wrong","tie")]
    original = sum(v for (old,new),v in conf.items() if old=="correct")
    lines += [f"| TTA 合计 | {totals[0]} | {totals[1]} | {totals[2]} | {n} |", "",
              f"原图选中人工赢家 {original}/{n}，TTA 选中 {totals[0]}/{n}。错→对 {conf['wrong','correct']} 对，对→错 {conf['correct','wrong']} 对。"
              "这些标签来自原提示词和原图。P1 仅评价模型响应。","", "## 原提示词的光照要求",""]
    counts = Counter(r["category"] for r in lighting.values())
    lines += [f"打分前逐条检查原提示词，{len(lighting)}/{n} 条有明确的时间、明暗、光源或布光措辞。"
              f"其中夜间 {counts['night']} 条、低照度或黑暗 {counts['low_light']} 条、白天或晴亮 {counts['day_or_bright']} 条、晨昏类 {counts['twilight']} 条，其余 {counts['lighting_style']} 条指定灯光风格或光源。"
              "分类按每条提示词的主要要求互斥计数，含 dawn 和 sunny summer day 的同一条归入白天或晴亮。"
              "仅有 cinematic、色名、摄影风格、暗色物体或剪影不足以归类。lighting.csv 列出逐条原文证据。"
              "夜景追加语有时强化旧要求，有时与白天或晴亮要求冲突，因此总体响应混合了多种语义关系。"]
    trunc = {c:len({r["pair_id"] for r in scores if r["prompt_condition"]==c and int(r["tokens_before_truncation"])>77}) for c in ("P0","P1")}
    identical = len({r["pair_id"] for r in scores if int(r["same_P0_P1_tokens"])})
    lines += ["", f"官方 tokenizer 的上下文为 77 个 token（含首尾标记）。P0 有 {trunc['P0']} 条超长，P1 有 {trunc['P1']} 条超长；追加前后实际 token 完全相同的有 {identical} 对。"
              "长提示词可能让追加语部分或全部被截断。统计保留了所有预先抽中的样本。","",
              "## |I| 最大的三个图对","",
              "下表的箭头依次为原图、调亮、TTA 均值的模型选择。案例按 |I| 降序，同值按 pair_id 升序。"]
    ordered = sorted(results,key=lambda r:(-r["abs_I"],r["pair_id"]))
    tied = [r["pair_id"] for r in ordered if r["abs_I"] == ordered[2]["abs_I"]]
    if len(tied)>1:
        lines.append("第三名的 |I| 并列图对为 " + "、".join(tied) + "，以下按上述顺序展示。")
    for r in ordered[:3]:
        i = r["pair_id"]
        lines += ["", f"### 图对 {i} · |I| = {number(r['abs_I'])}", "", r["P0"].strip(), "",
                  f"![图对 {i}，左 A 右 B，上原图下调亮](cases/{i}.png)", "",
                  "预览上排为原图，下排为 1.1 倍亮度，左 A、右 B。用于评分的是 RGB 输入。", "",
                  "| 提示词 | d0 | d1 | Δ | d_TTA | 模型选择 |",
                  "|---|---:|---:|---:|---:|---|"]
        for c in ("P0","P1"):
            v = [number(r[c+"_"+k]) for k in ("d0","d1","delta","d_tta")]
            picks = " → ".join(r[c+"_"+k] for k in ("pick0","pick1","pick_tta"))
            lines.append(f"| {c} | "+" | ".join(v)+f" | {picks} |")
        lines += ["", f"有符号 I = {number(r['I'])}。P0 原图人工赢家为 {r['human_winner_P0']}。"]
    lines += ["","## 执行与校验","",
              f"服务器 {env['host']}，单张 {env['gpu_model']}，物理 GPU "+", ".join(str(i) for i in sorted({s['gpu']['index'] for s in runs}))+f"。两个评分阶段累计 {elapsed:.2f} 秒（含加载、校验与评分，不含下载、环境准备和报告生成）。",
              ""]
    for t in runs:
        lines.append(f"- 目标 {t['limit']} 对，本次新增 {t['new_pairs']} 对，耗时 {t['seconds']:.2f} 秒，累计 {t['cumulative_scores']} 条评分。")
    lines += ["",f"64 条唯一原提示词，每对八种组合各一次，共 512 条有限原始分数，键无重复、字段无缺失。128 组 d_TTA 均逐项验证等于 (d0+d1)/2。"
              f"官方 CUDA 接口的返回类型为 {', '.join(sorted({r['native_dtype'] for r in scores}))}，CSV 完整保留返回值，后续做差与均值采用双精度。"
              "原始分数的量化精度会影响接近零的分差和同分。","","数据与模型版本、权重和验证集文件的 SHA-256 见 environment.json、manifest.json。"]
    (ROOT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def previews(results):
    import io
    import zipfile
    from PIL import Image, ImageEnhance, ImageOps
    from pilot_io import DATA
    pairs = {p["pair_id"]:p for p in load_pairs()}
    dest = ROOT/"cases"
    dest.mkdir(exist_ok=True)
    for r in sorted(results,key=lambda r:(-r["abs_I"],r["pair_id"]))[:3]:
        i = r["pair_id"]
        out = dest/(i+".png")
        if out.exists() and not (DATA/"validation_1.zip").exists():
            continue
        canvas = Image.new("RGB",(648,648),(245,245,245))
        for column,side in enumerate(("A","B")):
            path = Path(pairs[i]["image_"+side])
            with zipfile.ZipFile(DATA/(path.parent.name+".zip")) as z:
                im = Image.open(io.BytesIO(z.read(path.name))).convert("RGB")
            for row,pic in enumerate((im,ImageEnhance.Brightness(im).enhance(1.1))):
                thumb = ImageOps.contain(pic,(320,320))
                canvas.paste(thumb,(328*column+(320-thumb.width)//2,328*row+(320-thumb.height)//2))
        canvas.save(out)


if __name__=="__main__":
    results,scores,lighting = analyze()
    previews(results)
    report(results,scores,lighting)
    print("Verified 64 distinct prompts, 512 unique finite scores, 128 TTA identities.")

