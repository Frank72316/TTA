"""Plot real annotated examples and build the compact Chinese results page."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from design import HERE, SOURCE, rows, sha


def font(size):
    for path in ("C:/Windows/Fonts/msyh.ttc","/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/System/Library/Fonts/PingFang.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path,size)
    raise RuntimeError("Install Noto Sans CJK or point font() to an installed CJK font.")


def cases(root,photos,details,loc):
    sources={(r["pair_id"],r["side"]):r for r in rows(SOURCE/"images.csv")}
    chosen=[]
    for predicate in (lambda r:r["global4"]=="reversed" and r["local"]=="correct",lambda r:r["local"]=="reversed"):
        candidates=[r for r in details if predicate(r)]
        eligible=[r for r in candidates if all(sources[r["pair_id"],s]["license_id"] not in ("3","6") for s in "AB")]
        chosen.append((len(candidates),eligible[0] if eligible else None))
    panels=[]
    ink="#143240"; teal="#14635b"; blue="#00c5ff"; orange="#ff982d"
    for (count,r),title in zip(chosen,("全图分反，局部救回","局部仍分反")):
        if r is None:
            continue
        pid=r["pair_id"]
        panel=Image.new("RGB",(1100,278),"#f4f7f5")
        d=ImageDraw.Draw(panel)
        d.text((14,6),f'{pid}   {title}',font=font(22),fill=ink)
        d.text((479,10),f'蓝框 X = {r["x"]}     橙框 Y = {r["y"]}     白线 = 半幅分界',font=font(17),fill=ink)
        for j,side in enumerate("AB"):
            src=sources[pid,side]
            photo=photos/src["file"]
            assert sha(photo.read_bytes())==src["jpeg_sha256"]
            with Image.open(photo) as im:
                ox,oy,w,h=json.loads(src["valid_xywh"])
                tile=Image.new("RGB",(224,224),"#222b30")
                tile.paste(im.convert("RGB").resize((w,h),Image.Resampling.BICUBIC),(ox,oy))
            overlay=ImageDraw.Draw(tile)
            for k,col in (("x",blue),("y",orange)):
                overlay.rectangle(tuple(json.loads(src[k+"_input_xyxy"])),outline=col,width=2)
            x=ox+w/2
            overlay.line((x-1,oy,x-1,oy+h-1),fill="#45474b",width=1)
            overlay.line((x,oy,x,oy+h-1),fill="white",width=1)
            baseline="左" if side==r["true_left"] else "右"
            d.text((14+230*j,39),f'{side}  ·  X 在 Y {baseline}侧',font=font(16),fill=ink)
            panel.paste(tile.resize((205,205),Image.Resampling.LANCZOS),(14+230*j,65))
        d.text((479,53),"每张照片的实际局部分数",font=font(18),fill=teal)
        for n,head in enumerate(("候选","左 × X","左 × Y","右 × X","右 × Y","E")):
            d.text(([479,535,645,755,865,975][n],91),head,font=font(16),fill=ink)
        for idx,side in enumerate("AB"):
            c=loc[pid,side]
            y=126+idx*35
            d.text((479,y),side,font=font(18),fill=ink)
            for x,key in zip((535,645,755,865,975),("left_x","left_y","right_x","right_y","E")):
                d.text((x,y),f'{float(c[key]):+.6f}' if key=="E" else f'{float(c[key]):.6f}',font=font(17),fill=ink)
        d.line((479,197,1085,197),fill="#bccdca",width=1)
        d.text((479,209),f'真左图减真右图的证据差　全图 {float(r["global4_margin"]):+.6f}　局部 {float(r["local_margin"]):+.6f}',font=font(16),fill=teal)
        d.text((479,242),"正值分配正确，负值分配颠倒。色框仅用于展示。",font=font(15),fill=ink)
        panels.append(panel)
    if panels:
        result=Image.new("RGB",(1100,282*len(panels)-4),"white")
        for j,panel in enumerate(panels):
            result.paste(panel,(0,282*j))
        (root/"figures").mkdir(exist_ok=True)
        result.save(root/"figures/cases.png")
    return chosen


def smoke_sheet(root):
    ids=json.loads((root/"freeze.json").read_text())["smoke_pairs"]
    result=Image.new("RGB",(678,4*240+28),"#f4f7f5")
    draw=ImageDraw.Draw(result)
    for j,title in enumerate(("原图与有效画面中线","左半幅 · 实际输入","右半幅 · 实际输入")):
        draw.text((226*j+8,1),title,font=font(16),fill="#143240")
    for j,(pid,side) in enumerate((i,s) for i in ids for s in "AB"):
        for k,v in enumerate(("mid","left","right")):
            with Image.open(root/"smoke"/f"{pid}_{side}_{v}.png") as im:
                result.paste(im,(226*k,26+240*j))
        draw.text((9,247+240*j),f"{pid} {side}",font=font(14),fill="#143240")
    result.save(root/"smoke/overview.jpg",quality=94)


def present(root,photos):
    stats=json.loads((root/"analysis.json").read_text())
    details=rows(root/"pair_results.csv")
    loc={(r["pair_id"],r["side"]):r for r in rows(root/"image_results.csv")}
    table={(r["clear_images_in_pair"],r["method"]):r for r in rows(root/"summary.csv")}
    moves={(r["reference"],r["before"],r["after"]):int(r["n"]) for r in rows(root/"transitions.csv")}
    noun={r["localization"]:r for r in rows(root/"localization_summary.csv")}
    runs=json.loads((root/"runs.json").read_text())
    first,nextcase=cases(root,photos,details,loc)
    smoke_sheet(root)
    net=stats["net_pairs"]
    correct=lambda group,st:int(table[str(group),st]["correct"])
    gains=[correct(g,"local")-correct(g,"global4") for g in (2,1,0)]
    n=stats["clear_photos"]
    lines=["# 固定左右半图，能否帮助 HPS 判断物体的左右关系？","",
        f'**1. 同预算下，局部名词方案净多分对 {net} 对。** 每张照片均用两个视觉输入、四条图文评分。局部方案分对 {correct("all","local")}/64，全图加镜像分对 {correct("all","global4")}/64，严格正确率差 **{stats["diff_pp"]:+.1f} 个百分点**（整图对配对 bootstrap 95% 区间 [{stats["lo95_pp"]:+.2f}, {stats["hi95_pp"]:+.2f}]）。',"",
        f'**2. 收益在两张图都清楚分居时最大，也出现在其余两层。** 两张、仅一张、零张满足分居标准的图对分别净增 **{gains[0]}、{gains[1]}、{gains[2]} 对**。标准在评分前冻结，要求每个物体至少 75% 的可见框面积落在相反半边。',"",
        f'**3. 在 {n} 张清楚分居的照片中，两个名词都定位正确 {noun["both_correct"]["n"]} 张**，仅一个正确 {noun["one_correct"]["n"]} 张、两个都错 {noun["both_wrong"]["n"]} 张、含同分 {noun["has_tie"]["n"]} 张（任一同分优先单列）。X 在正确半边得分更高 {stats["x_localized"]}/{n}，Y 为 {stats["y_localized"]}/{n}。仅定位对一个名词时，合并证据仍有 {noun["one_correct"]["E_correct"]}/{noun["one_correct"]["n"]} 张指向正确方向。',"",
        "| 方向证据 | 每图视觉输入／图文评分 | 正确 | 颠倒 | 同分 |","|---|---:|---:|---:|---:|"]
    for st,name,budget in (("original","原图左右两句分差","1 / 2"),("global4","全图＋镜像的方向分差","2 / 4"),("local","左右半图的名词分差","2 / 4")):
        r=table["all",st]
        lines.append(f'| {name} | {budget} | {r["correct"]} | {r["reversed"]} | {r["tie"]} |')
    lines+=["",f'相对四评分全图对照，局部方案把 **{moves["global4","reversed","correct"]} 对颠倒和 {moves["global4","tie","correct"]} 对同分变为正确**，也把 **{moves["global4","correct","reversed"]} 对原先正确的图对分反**。相对原图两评分，保住 {moves["original","correct","correct"]}/30 对，救回 {moves["original","reversed","correct"]}/34 对。[逐对转移与 ID](transitions.csv)',"",
        f'128 张照片中，清楚分居 {n} 张，其余 {stats["other_photos"]} 张。下表以自然图对为单位。',"",
        "| 每对清楚分居的照片数 | 图对数 | 原图正确 | 全图＋镜像正确 | 局部正确 | 正确率净差 |","|---|---:|---:|---:|---:|---:|"]
    for g in ("2","1","0"):
        den=int(table[g,"local"]["n"])
        lines.append(f'| {g} | {den} | {correct(g,"original")}/{den} | {correct(g,"global4")}/{den} | {correct(g,"local")}/{den} | {(correct(g,"local")-correct(g,"global4"))/den*100:+.1f} 个百分点 |')
    lines+=["","每对把证据较大的照片分给左题。令 d 为同一视图左提示分减右提示分，全图对照用 [d(原图)−d(镜像)]/2。局部证据 E 为 X 的左半图分减右半图分，加上 Y 的右半图分减左半图分。两者共用同一分配规则。图像裁法和文本也同时改变，这批结果支持组合方案，尚不能分离两个改动各自的贡献。",""]
    if first[1] or nextcase[1]:
        lines+=["![纠正与颠倒案例：两张照片的中线、物体框与四个局部原始分数](figures/cases.png)",""]
    lines+=[f'全图分反、局部救回共 {first[0]} 对，局部分反共 {nextcase[0]} 对。图例按类别内可公开展示裁图的最小 ID 选取。[照片来源](credits.md) · [四项原始评分](scores.csv) · [逐图定位](image_results.csv) · [逐对分配](pair_results.csv)',
        "",f'Asus RTX 4090 · HPS v2.1，CUDA FP16。新增 {sum(r["new_scores"] for r in runs)} 条评分、{sum(r["new_visual_forwards"] for r in runs)} 次视觉前向；续跑命中 {sum(r["checkpoint_score_hits"] for r in runs)} 条冒烟评分，另读取旧缓存 512 条。两段评分墙钟合计 {sum(r["wall_seconds"] for r in runs):.1f} 秒，图像前向及相似度合计 {sum(r["image_scoring_seconds"] for r in runs):.1f} 秒。[复现与坐标映射](README.md) · [两对冒烟输入](smoke/overview.jpg)']
    (root/"report.md").write_bytes(("\n".join(lines)+"\n").encode())
    return [r["pair_id"] if r else None for _,r in (first,nextcase)]


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=HERE)
    ap.add_argument("--photos",type=Path,required=True)
    a=ap.parse_args(); print("Cases",present(a.out,a.photos))
