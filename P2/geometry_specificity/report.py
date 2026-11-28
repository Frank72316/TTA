"""Build the Chinese report from the scored development and calibration cohorts."""
import argparse
import json
from pathlib import Path
from statistics import median

from history import HERE, read


def table(header,rows):
    return ["","| "+" | ".join(header)+" |","|"+"---|"*len(header),
            *["| "+" | ".join(map(str,r))+" |" for r in rows],""]


def make(root):
    main=root/"dev256"
    load=lambda n:read(main/(n+".csv"))
    stats=load("full_strategy_summary")
    contrasts=load("full_comparison")
    coverage=load("full_coverage")
    cases=load("seven_cases")
    per=load("full_seven_parameters")
    cal=read(root/"old64/full_strategy_summary.csv")
    matched=json.loads((root/"strength_freeze.json").read_text())
    counter=lambda a:sum(int(r["correct"])>0 and float(r["m0_new"])<0 for r in a)
    found=[counter([r for r in per if r["K"]==str(k) and r["strategy"]=="S"]) for k in (3,5,9)]
    ts=next(r for r in coverage if r["K"]=="9" and (r["left"],r["right"],r["threshold"])==("T","S","any"))
    def diff(r):
        return f'{float(r["diff_pp"]):+.3f} [{float(r["lo_pp"]):+.3f}, {float(r["hi_pp"]):+.3f}]'
    main_test=[r for r in contrasts if r["K"]=="9" and r["left"]=="TS"]
    conclusion=("两项区间下端都高于零，本开发集上混合有净优势。" if all(float(r["lo_pp"])>0 for r in main_test) else
                "本开发集尚未显示混合稳定超过两种单类。")
    if float(main_test[0]["hi_pp"])<0 and float(main_test[1]["lo_pp"])<=0<=float(main_test[1]["hi_pp"]):
        conclusion="在这批开发提示词上，均匀混合的严格选对率低于纯平移，相对纯缩放的区间跨零。"
    lines=["# 轻微平移与中心缩放能纠正同一批图对吗？","",
           f'① 在原来的七对上，缩放在 K=3、5、9 下分别能纠正 **{found[0]}、{found[1]}、{found[2]} 对**（至少一套严格选对、且本轮原图选错）。',
           "",f'② 在完整 256 对上，K=9 纯平移独有 **{ts["left_only"]} 对**，纯缩放独有 **{ts["right_only"]} 对**，共有 **{ts["both"]} 对**。两种纯策略各只有一套，至少一次与至少 25% 的覆盖完全相同。',
           "",f'③ K=9 混合相对纯平移、纯缩放的严格选对率差分别为 **{diff(main_test[0])}**、**{diff(main_test[1])}** 个百分点（95% 配对区间）。'+conclusion,
           "","## 七个历史案例","",
           "这 256 对沿用 P3 的开发样本。七例的入选依据来自 P2/translation 的三次评分历史缓存：原图严格选错，BB、CC、BC 没有搭配选对，BT 至少一套选对。我们在新缩放评分前从双方原始分数重新枚举并冻结。历史 B、C 各用 0.90、0.95、1.05、1.10，T 用四方向 3%；另存 P3 在 L4 上的八方向预算结果供对照。"]
    lines+=table(["图对","人工赢家","旧 BT /16","旧 TT /6","T，K=3","S，K=3","T，K=5","S，K=5","T，K=9","S，K=9"],
                 [[r["pair_id"],r["human"],r["BT"],r["TT"],*[r[f+"_K"+str(k)] for k in (3,5,9) for f in "TS"]] for r in cases])
    for k in (3,5,9):
        lost=[r["pair_id"] for r in per if r["K"]==str(k) and r["strategy"]=="T" and (int(r["correct"])==0 or float(r["m0_new"])>=0)]
        lines += [f'K={k} 时，新平移池未纠正的历史案例为 '+("、".join(lost) or "0 对")+"。",""]
    align=root/"baseline_alignment.csv"
    if align.exists():
        control=read(align)
        focus=[r for r in control if r["historical_seven"]=="1"]
        lines += [f'历史前向为 CUDA AMP FP16，本次原图和四种增强均以同一生产配置重算。256 对中原图判定（正／零／负）有 {sum(int(r["baseline_changed"]) for r in control)} 对不同。用本轮分数只取历史四个 3% 方向，三次纯平移在七例中的命中数如下，可直接检查推理数值与换池的影响。']
        lines += table(["图对","历史 TT /6","同四方向本轮 TT /6","本轮同分 /6"],
                       [[r["pair_id"],r["historical_TT_correct"],r["current_3pct_TT_correct"],r["current_3pct_TT_tie"]] for r in focus])
    lines+=["再单看原图＋一个几何版本，下面列出能直接使原始错例选对的方向或倍率。某个版本也可能只在搭配里有用；[完整参与次数](dev256/full_seven_parameters.csv)记录每个版本进入成功组合的次数、版本本身的分差和相对原图的位移方向。"]
    lines+=table(["图对","平移方向／百分比","缩放倍率"],[[i,*[next(r["two_score_correct"] for r in per if r["pair_id"]==i and r["K"]=="3" and r["strategy"]==f) or "无" for f in "TS"]] for i in [c["pair_id"] for c in cases]])
    lines+=["三次预算给出另一种可读的方向诊断：固定其中一个增强版本，另一个在其余七档中均匀选择。下表列出与最多成功搭配相伴的版本；并列全部列出。该频率由两档与原图共同决定。"]
    frequent=[]
    for i in [c["pair_id"] for c in cases]:
        line=[i]
        for f in "TS":
            item=next(r for r in per if (r["pair_id"],r["K"],r["strategy"])==(i,"3",f))
            freq=json.loads(item["correct_combo_inclusions"])
            best=max(freq.values())
            line.append(("、".join(v for v,x in freq.items() if x==best)+f"（{best}/7）") if best else "无（0/7）")
        frequent.append(line)
    lines+=table(["图对","平移最高条件频率","缩放最高条件频率"],frequent)
    lines+=["## 完整 256 对的同预算比较","",
            f'原图基线严格选对 {stats[0]["base_correct"]} 对、选错 {stats[0]["base_wrong"]} 对、同分 {stats[0]["base_tie"]} 对。B、C、T、S 分别指亮度、对比度、平移、中心缩放，TS 的两个变换分别作用于原图。每一套从八个版本中无放回取样，A/B 共用参数。每张图都含一次原图，图像分数等权取均值，再按人工赢家相对另一张图的分差符号判定。下表的纠正、改错与同分为均匀取一套时的期望对数。']
    lines+=table(["K","策略","搭配数","严格选对率","错→对","对→错","对→同分","错→同分","原同分→对／错／同分"],
                 [[r["K"],r["strategy"],r["combinations"],f'{float(r["accuracy_pct"]):.3f}%',
                   *[f'{float(r[k]):.3f}' for k in ("wrong_to_correct","correct_to_wrong","correct_to_tie","wrong_to_tie")],
                   "/".join(f'{float(r["tie_to_"+k]):.3f}' for k in ("correct","wrong","tie"))] for r in stats])
    lines+=table(["K","左 − 右","搭配数 左 / 右","严格选对率差 [95% 区间]，百分点"],
                 [[r["K"],r["left"]+" − "+r["right"],r["left_combos"]+" / "+r["right_combos"],diff(r)] for r in contrasts])
    lines+=["整对提示词 bootstrap 10,000 次，固定种子 20261128，区间为逐比较描述性区间。对于每个图对，每个版本在均匀组合中等概率入选。混合均值等于两纯均值的一半之和，因为两族分配的槽位相等且原图权重相同。主批全部预算的最大绝对核验残差为 "+f'{max(float(r["mix_identity_worst"]) for r in contrasts):.3g} HPS 分。',"",
            "以下只取原始错例；每格为左独有／共有／右独有。[有向覆盖明细](dev256/full_coverage.csv)保留全部 ID。"]
    lines+=table(["K","左 / 右","两侧搭配总数","至少一次","至少 25%"],[
        [r["K"],r["left"]+" / "+r["right"],r["left_combos"]+" / "+r["right_combos"],*[
        "/".join(z[key] for key in ("left_only","both","right_only")) for z in coverage
        if (z["K"],z["left"],z["right"])==(r["K"],r["left"],r["right"])]] for r in coverage if r["threshold"]=="any"])
    lines+=["稳定事件按至少 25% 的搭配严格纠正或严格改错，下列 ID 只统计提示词一次。"]
    lines+=table(["K","方法","稳定纠正","稳定改错"],[[r["K"],r["strategy"],r["fix_stable_ids"] or "无",r["harm_stable_ids"] or "无"] for r in stats if r["strategy"] in ("T","S")])
    t9,s9=[next(r for r in stats if (r["K"],r["strategy"])==("9",f)) for f in "TS"]
    lines+=[f'主预算下，平移独有纠正 {ts["left_only_ids"] or "无"}，缩放独有纠正 {ts["right_only_ids"] or "无"}，共有纠正 {ts["both_ids"] or "无"}。']
    if t9["harm_stable_ids"]==s9["harm_stable_ids"] and t9["harm_stable_ids"]:
        lines += [f'这两种纯策略把同一批原本选对的图改错：{t9["harm_stable_ids"]}。']
    lines+=["","## 用旧 64 对检查扰动强度","",
            "校准集和开发集分别统计。每个版本先计算 64 对的绝对分差位移中位数，以及 128 张官方模型输入的平均像素绝对变化的中位数（反归一化后 0–255 单位）。只在两族分差位移中位数的交集内，一对一匹配尽可能多的版本，并最小化总绝对距离。随后冻结匹配文件再开始 256 对打分。"]
    devstrength={r["view"]:r for r in load("version_strength")}
    indexes={r["view"]:r for r in matched["version_statistics"]}
    lines+=table(["匹配版本 T / S","64 对位移 T / S","64 对像素变化 T / S","256 对位移 T / S","256 对像素变化 T / S"],[
        [a+" / "+b,*[" / ".join(f'{float(bank[v][field]):.5f}' for v in (a,b)) for bank in (indexes,devstrength) for field in ("median_abs_margin_shift","median_input_rgb255")]] for a,b in matched["pairs"]])
    lines+=[f'共同分差位移范围下端 {matched["common_low"]:.6f}、上端 {matched["common_high"]:.6f}，保留 {len(matched["pairs"])} 对版本。旧 64 对完整池 K=9 严格选对率为 '+ "、".join(f'{r["strategy"]} {float(r["accuracy_pct"]):.3f}%' for r in cal if r["K"]=="9")+"。",""]
    if matched["pairs"]:
        paired=load("matched_comparison")
        lines+=table(["匹配池 K","比较","严格选对率差 [95% 区间]，百分点"],[[r["K"],r["left"]+" − "+r["right"],diff(r)] for r in paired])
        last=max(int(r["K"]) for r in paired if r["left"]=="T")
        comp=next(r for r in paired if r["K"]==str(last) and r["left"]=="T")
        full=next((r for r in contrasts if r["K"]==str(last) and r["left"]=="T"),None)
        event=next(r for r in load("matched_coverage") if r["K"]==str(last) and r["left"]=="T" and r["threshold"]=="quarter")
        if full:
            lines+=["在开发集相同的 K="+str(last)+" 下，T−S 严格选对率差从完整池的 "+diff(full)+" 变为匹配池的 "+diff(comp)+" 个百分点。"]
        lines+=[f'匹配池 K={last} 的稳定纠正分别有平移独有 {event["left_only"]} 对、缩放独有 {event["right_only"]} 对、两者共有 {event["both"]} 对。']
        gaps=[abs(float(devstrength[t]["median_input_rgb255"])-float(devstrength[s]["median_input_rgb255"])) for t,s in matched["pairs"]]
        lines += [f'开发集匹配版本之间的像素变化中位数差的绝对值，跨匹配取中位为 {median(gaps):.3f}/255。']
        lines+=["匹配约束作用于这批图的典型分差位移。像素变化和逐图位移仍有差异，无法把平移多纠正的全部案例归因于算子类型。"+("本次匹配池每族少于八档，K=9 的纯策略只能引用完整池结果。" if len(matched["pairs"])<8 else "")]
    else:
        lines+=["两族缺少共同支持，无法从本次结果分离算子类型与扰动强度。"]
    lines+=["","## 模型实际看到的案例",""]
    records=read(root/"case_figure_data.csv")
    if any(int(r["count"]) for r in records):
        lines+=["![原图与触发视图](figures/cases.jpg)",""]
    lines+=["类别按 K=9 纯策略严格决策选择，每类优先取编号最小且尚未展示的图对。图中每种变换均并列 A/B，展示选定方法中朝相应判定移动最多的单版本。K=9 的最终分差另列；单个版本只贡献八个增强分数之一。所有展示输入均与评分记录的张量哈希逐张匹配。"]
    lines+=table(["类别","实际例数","图对／人工赢家","原图分差","单个展示视图的有向分差","全池 T9 / S9"],[
        [dict(T_only="仅平移纠正",both="两者均纠正",harm="几何扰动改错")[r["kind"]],r["count"],(r["pair_id"]+" "+r["human"]).strip() or "—",r["m0"] or "—",
         "，".join(f"{v} {float(m):+.6f}" for v,m in zip(r["shown"].split(),r["shown_margins"].split())) or "—",
         r["T9"]+" / "+r["S9"] if r["pair_id"] else "—"] for r in records])
    lines+=["## 运行与完整性",""]
    totals=[]
    for batch,n in (("old64",64),("dev256",256)):
        run=json.loads((root/batch/"runs.json").read_text())
        raw=read(root/batch/"scores.csv")
        dup=sum(r["scored_view"]!=r["view"] for r in raw)
        totals.append([n,len(raw),sum(r["new"] for r in run),sum(r["checkpoint_hits"] for r in run),
                       dup,sum(r["external_hits"] for r in run),f'{sum(r["image_inference_s"] for r in run):.1f}'])
    lines+=table(["提示词","原始分数","新增图像前向","断点命中","重复张量复用","外部缓存命中","各工作进程图像前向秒之和"],totals)
    complete=json.loads((root/"completed.json").read_text())
    lines += [f'生产流程实际墙钟 {complete["elapsed_s"]:.1f} 秒，设备 {complete["device"]}，并行进程 {complete["shards"]}，图像批大小 {complete["batch"]}。各批分别用原图标签；平局保留在分母中。这里是对既有开发提示词的观察。']
    qa=HERE/"qa/smoke_comparison.json"
    if qa.exists() and complete["device"]=="cpu":
        smoke=json.loads(qa.read_text())
        aux=smoke["singleton_runs"][0]
        lines+=["",f'生产运行之外，CPU 单张冒烟另做 {aux["new"]} 次图像前向，墙钟 {aux["wall_s"]:.1f} 秒。正式 CPU 批评分与逐张冒烟的 {smoke["matching_actual_tensor_hashes"]} 个输入哈希逐一相同，最大绝对分差异为 {smoke["max_abs_score_diff"]:.3g}。含这次校验的实际新图像前向合计 {sum(x[2] for x in totals)+aux["new"]} 次。生产结果采用同一次实验的权重、预处理和精度；每张原图的全部策略使用同一原图分数。']
    if complete["device"]=="cuda":
        env=json.loads((root/"dev256/environment.json").read_text())
        lines+=["",f'正式结果在 Asus4090x8 的物理 2 号卡 {env["hardware"]} 上完成，驱动 580.159.04，PyTorch {env["torch"]}，{env["precision"]}，官方 HPS v2.1 验证预处理。GPU 合计新图像前向 {sum(x[2] for x in totals):,} 次，断点复用 {sum(x[3] for x in totals)} 条。上表最后一列只计图像前向。生产墙钟另外包含权重校验、模型加载、两批冒烟、强度冻结与离线枚举。']
        checks=HERE/"qa/asus_smoke_checks.json"
        if checks.is_file():
            smoke=json.loads(checks.read_text())
            lines+=["",f'两对冒烟在 CPU 与 Asus 上的 {smoke["matching_CPU_and_GPU_tensor_sha"]} 个官方输入张量哈希全部相同。与历史缓存可按 RGB 逐像素比对的 {smoke["matched_historical_rgb"]} 项中，{smoke["old_gpu_score_exact"]} 项分数相同，最大绝对差 {smoke["old_gpu_max_score_diff"]}。所有正式评分都在 Asus 上产生。']
        audit=root/"verification.json"
        if audit.is_file():
            v=json.loads(audit.read_text())
            lines+=["",f'另将原始分数无损换算为整数，独立核验 {sum(r["integer_combination_decisions"] for r in v):,} 个图对×策略搭配的严格决策，差异 {sum(r["decision_mismatches"] for r in v)} 项。']
            if all("unique_tensor_hashes" in r for r in v):
                lines+=["逐批跨图像检查输入哈希。"+ "。".join(f'{r["pairs"]} 对中，{r["scores"]:,} 条评分对应 {r["unique_tensor_hashes"]:,} 个不同图像张量，{r["unique_tensor_token_pairs"]:,} 个不同的张量与提示词 token 组合' for r in v)+"。"]
        extra=root/"auxiliary_cpu/cpu_superseded.json"
        if extra.is_file():
            r=json.loads(extra.read_text())
            lines+=["",f'切换到 Asus 前，另一台机器的 CPU 探查留有 64 对校准、{r["completed"]["dev256"]["checkpointed_pairs"]} 对开发数据的完整检查点，以及单独的 132 次单张冒烟。其 FP32 分数与耗时单独留档。停止时在途前向未逐次记账；上述正式 GPU 统计完整。']
    (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--root",type=Path,required=True)
    make(cli.parse_args().root)
