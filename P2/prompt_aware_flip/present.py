"""Write the case figure and Chinese report from frozen manifests and scored results."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from analyze import analyze, NAMES, STRATEGIES
from common import HERE, rows, save_rows


def font(size):
    for name in ("DejaVuSans.ttf","Arial.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def table(headers,body):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                      *["| "+" | ".join(map(str,r))+" |" for r in body]])


def generate(out,views):
    res,summary,contrasts=analyze(out)
    pairs={p["pair_id"]:p for p in rows(out/"pairs.csv")}
    by_id={r["pair_id"]:r for r in res}
    imgs={(s["pair_id"],s["side"]):s for s in rows(out/"images.csv")}
    freeze=json.loads((out/"freeze.json").read_text())
    original_lost=[r["pair_id"] for r in res if r["original_both"] and not r["image_only_both"]]
    restored=[i for i in original_lost if by_id[i]["synchronized_both"]]
    sync_failed=[r["pair_id"] for r in res if not r["synchronized_both"]]
    sync_failed.sort(key=lambda i:(not by_id[i]["image_only_both"],i))
    selection=[]
    for group in (restored,original_lost,sync_failed):
        publishable=[i for i in group if all(imgs[i,s]["license_id"] not in ("3","6") for s in "AB")]
        if publishable and not any(i in group for i in selection):
            selection.append(publishable[0])
    figdir=out/"figures"; figdir.mkdir(exist_ok=True)
    panels=[]
    for i in selection:
        p=pairs[i]; r=by_id[i]
        canvas=Image.new("RGB",(1024,487),"#f7f8f5")
        draw=ImageDraw.Draw(canvas); ink="#172a38"
        draw.text((20,12),f'{i}   |   Blue: {p["x"]}     Orange: {p["y"]}',font=font(22),fill=ink)
        for j,(side,v) in enumerate((("A","O"),("B","O"),("A","F"),("B","F"))):
            x=20+j*249
            label=("Original " if v=="O" else "Mirrored ")+side
            draw.text((x,52),label,font=font(17),fill=ink)
            with Image.open(views/f"{i}_{side}_{v}.png") as im:
                assert im.size==(224,224)
                canvas.paste(im,(x,78))
            src=imgs[i,side]
            ox,_,w,_=json.loads(src["valid_xywh"])
            for col,colour in (("x","#26baff"),("y","#ff9822")):
                x1,y1,x2,y2=json.loads(src[col+"_input_xyxy"])
                if v=="F":
                    x1,x2=2*ox+w-x2,2*ox+w-x1
                draw.rectangle((x+x1,78+y1,x+min(223,x2),78+min(223,y2)),outline=colour,width=1)
        y=320
        for q in ("left","right"):
            draw.text((20,y),p["prompt_"+q],font=font(19),fill=ink)
            draw.text((625,y),"Box answer:  "+p["winner_"+q],font=font(19),fill=ink)
            y+=30
        for j,(st,title) in enumerate(zip(STRATEGIES,("Original","Image-only TTA","Synchronized TTA"))):
            x=20+330*j
            draw.text((x,402),title,font=font(18),fill=ink)
            draw.text((x,433),f'Left: {r[st+"_left"]}     Right: {r[st+"_right"]}',font=font(20),
                      fill="#067148" if r[st+"_both"] else "#944019")
        panels.append(canvas)
    if panels:
        full=Image.new("RGB",(1024,487*len(panels)))
        for j,im in enumerate(panels):
            full.paste(im,(0,j*487))
        full.save(figdir/"cases.png")
    def comp(after,before):
        return next(d for d in contrasts if (d["after"],d["before"],d["metric"])==(after,before,"both"))
    def ci(d):
        return f'{d["diff_pp"]:+.3f} 个百分点（95% 区间 [{d["lo95_pp"]:+.3f}, {d["hi95_pp"]:+.3f}]）'
    c=comp("image_only","original"); s=comp("synchronized","image_only"); o=comp("synchronized","original")
    overall={v["strategy"]:v for v in summary}
    n=len(res); k=2*n
    ids=lambda items:"、".join(items) if items else "无"
    def wins(st):
        return [r["pair_id"] for r in res if r[st+"_both"]]
    pct=lambda x:f'{100*x:.2f}%'
    transitions=[]
    for before,after in (("original","image_only"),("image_only","synchronized"),("original","synchronized")):
        transitions.append(dict(before=before,after=after,wrong_to_right=0,right_to_wrong=0))
        for r in res:
            for q in ("left","right"):
                b,a=r[before+"_"+q],r[after+"_"+q]
                correct=r["winner_"+q]
                transitions[-1]["wrong_to_right"]+=b not in (correct,"=") and a==correct
                transitions[-1]["right_to_wrong"]+=b==correct and a not in (correct,"=")
    save_rows(out/"question_transitions.csv",transitions)
    raw={(r["image_id"],r["view"],r["direction"]):float(r["score"]) for r in rows(out/"scores.csv")}
    changed=[sum(raw[i,v,"left"]!=raw[i,v,"right"] for i,view,q in raw if q=="left" and view==v) for v in ("O","F")]
    start=[
        "# 翻图时交换 left/right，能让 HPS 更会判断左右吗？",
        f'① 以同一对照片的左右两题都严格答对为准，只翻图使原有成功丢失 **{c["pairs_decreased"]} 对**，新增 **{c["pairs_improved"]} 对**，成功数从 {overall["original"]["both_count"]}/{n} 变为 {overall["image_only"]["both_count"]}/{n}。',
        f'② 相对只翻图，同步交换方向词救回 **{s["pairs_improved"]} 对**，又使 **{s["pairs_decreased"]} 对**失去双题全对，主指标相差 **{ci(s)}**。',
        f'③ 与原图相比，同步策略的双题全对率差为 **{ci(o)}**。'+("本批未观察到净收益。" if o["diff_pp"]<=0 else "本批点估计提高。"),
        "## 三种策略在同一批图上的选择",
        f'每道题在两张照片间选一张，真值来自 COCO 实例框。交换方向词会把正确候选从 A 换成 B 或从 B 换成 A。主指标的分母是 {n} 个完整图对，单题指标的分母是 {k} 题。下表的“仍选同图”指两道题都严格选择了同一个候选。',
        table(["评分策略","双题全对／图对","单题严格选对","单题严格选错","单题同分","换方向仍选同图"],[
            [NAMES[t["strategy"]],f'{t["both_count"]}/{n} ({pct(t["both_rate"])})',
             f'{t["question_accuracy_count"]}/{k} ({pct(t["question_accuracy_rate"])})',
             f'{t["question_wrong_count"]}/{k}',f'{t["question_tie_count"]}/{k} ({pct(t["question_tie_rate"])})',
             f'{t["same_choice_count"]}/{n} ({pct(t["same_choice_rate"])})'] for t in summary]),
        "双题均选对的 ID 按原图、只翻图、图文同步依次为 **"+ids(wins("original"))+"**、**"+ids(wins("image_only"))+"**、**"+ids(wins("synchronized"))+"**。"
        +f'同步策略在 {overall["synchronized"]["same_choice_count"]}/{n} 对中两次都选了同一张照片。由于左右两题答案相反，在这些图对上会恰好答对一题。',
        f'方向词改变了 {changed[0]}/{k} 张原图、{changed[1]}/{k} 张镜像图的单图分数。模型对用词有数值响应，但该响应很少足以改变两张自然照片的相对排序。下表把每道题的纠正、改错单独计数。',
        table(["策略变化","单题 错→对","单题 对→错"],[[NAMES[r["before"]]+" → "+NAMES[r["after"]],r["wrong_to_right"],r["right_to_wrong"]] for r in transitions]),
        "区间使用种子 42 的 10,000 次配对 bootstrap。每次抽取完整图对，保留两道题和三种策略。稀少的成功或失败事件限制了对总体差异的判断。[各指标差值与区间](comparisons.csv) · [逐对选择与分差](pair_results.csv)",
        "## 选图与评分",
        f'COCO 2017 validation 的 {freeze["bidirectional_category_pairs"]} 种类别组合有两个方向的合格照片。固定种子 42，打乱类别组合后轮流取对，打乱同组照片顺序以及主体 X 和 A/B 的分配。本批最终为 {n} 对、{2*n} 张不同照片、{freeze["category_pairs"]} 种不同类别组合。每张图的两个目标类别分别只有一个标注实例且均为非 crowd。同一目标类别带额外 crowd 标注的照片也剔除。每对左右提示只替换一个方向词。照片和两个物体的先后顺序由标注确定，抽样在模型评分前完成。',
        '实际 HPS 验证预处理将长边缩放至 224 并居中补边。框按实际整数缩放尺寸分别映射横纵轴，裁到内容区后可见面积至少 80%，两框在水平方向不重叠，中心横距至少为内容区宽度的 20%。逐图用常量画布经官方 resize 测出内容区，核对镜像前后左右标签互换。对应的英文原句、框坐标、文件哈希与带框预览均已保存。[冻结清单](pairs.csv) · [照片和框](images.csv) · [重建带框预览的命令](README.md#从-coco-标注和原图复现)',
        '原图和从 RGB 原图生成的水平镜像各算左、右两条提示。原图策略直接比较两张原图的分数。只翻图策略平均当前题的原图和镜像分数。同步策略平均当前题的原图分数和相反方向题的镜像分数。所有策略用 A 减 B 的分差判定，原生 FP16 分数转 float64 求和，严格为零记为同分。这里评估的是框定义的空间关系，COCO 标注没有评价照片质量或审美偏好。',
        "## 用自身镜像检查实现",
        f'固定种子从 {2*n} 张已评分照片中不放回抽 8 张，用每张照片的自身镜像作候选 B。任一提示 q 下，两侧只翻图的结果分别是 (原图分＋镜像分)/2 与 (镜像分＋原图分)/2。缓存重算的 **8 对 × 2 题全部精确同分**，最大绝对分差为 0。主实验的两个候选来自不同照片，所以这个恒等式只适用于单列检查。[16 项检查](mirror_check.csv)',
        "## 案例",
        f'“原图双题全对而只翻图失败”共 {len(original_lost)} 对，“同步改词后恢复这些原有成功”共 {len(restored)} 对。同步策略未获得双题全对的有 {len(sync_failed)} 对。案例按这三类依次覆盖，失败类优先展示只翻图能答对两题的图对，再按 ID 取首个可发布镜像的例子。',
    ]
    if panels:
        start+=["![两张模型输入及其镜像、左右两题与实际选择](figures/cases.png)",
                "蓝框标主体，橙框标参照物。框仅用于读图，评分时输入没有标记。"]
    for i in selection:
        p=pairs[i]; r=by_id[i]
        start += [f'**{i} · {p["x"]} 与 {p["y"]}。** 左题答案 {p["winner_left"]}，右题答案 {p["winner_right"]}。下列分差均为 A−B，正值选择 A。',
             table(["实际提示","原图 选择／分差","只翻图 选择／分差","图文同步 选择／分差"],[
                 [p["prompt_"+q],*[f'{r[st+"_"+q]} ／ {r[st+"_margin_"+q]:+.8f}' for st in STRATEGIES]] for q in ("left","right")])]
    start+=["[案例照片署名与许可](photo_credits.md)","## 适用范围与运行记录",
        '结论限于这批自然照片、冻结的提示句式和 HPS v2.1。两张自然照片在场景与外观上也有差异，结果能检验方向词能否改变模型的最终选择，尚不足以辨别模型在依据哪些视觉特征打分。实例框能确定标注目标的相对位置，目标在 224 像素输入中的大小和辨识难度仍有差别。']
    env=json.loads((out/"environment.json").read_text()); run=json.loads((out/"runs.json").read_text())[0]
    start += [f'正式运行使用 Asus 的 NVIDIA RTX 4090、PyTorch {env["torch"]}、HPS v2.1 和 CUDA autocast FP16。先验证两对的 token、翻转方向、补边以及赢家映射，再完成全体。首次运行得到 {run["new_image_text_scores"]} 条图文分数，共 {run["new_image_forwards"]} 次图像编码，断点命中 {run["checkpoint_scores"]} 条。评分墙钟 {run["wall_seconds"]:.1f} 秒，计入 GPU 图像前向及两次相似度计算的时间合计 {run["gpu_image_seconds"]:.1f} 秒。[原始评分](scores.csv) · [冒烟检查](smoke.json) · [复跑方法](README.md)']
    (out/"report.md").write_bytes(("\n\n".join(start)+"\n").encode())
    return selection


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=HERE)
    ap.add_argument("--views",type=Path,required=True)
    a=ap.parse_args(); print("Illustrated",generate(a.out,a.views))
