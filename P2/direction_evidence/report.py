"""Render the Markdown handoff and, optionally, its one-page PDF using a local browser."""
import argparse
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from analyze import HERE, SOURCE, rows, sha


def md_table(headers, body):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                      *["| "+" | ".join(map(str,r))+" |" for r in body]])


def html_table(headers, body):
    cells = lambda row,tag:"<tr>"+"".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in row)+"</tr>"
    return "<table><thead>"+cells(headers,"th")+"</thead><tbody>"+"".join(cells(r,"td") for r in body)+"</tbody></table>"


def case_files(out,selected,photos):
    images = {(r["pair_id"],r["side"]):r for r in rows(SOURCE/"images.csv")}
    folder=out/"photos"; folder.mkdir(exist_ok=True)
    for case in selected:
        for side in "AB":
            src=images[case["pair_id"],side]
            dest=folder/src["file"]
            if not dest.exists():
                if not photos:
                    raise FileNotFoundError(f"{dest} missing. Supply --photos with the cached COCO val2017 directory.")
                original=photos/src["file"]
                assert sha(original.read_bytes())==src["jpeg_sha256"]
                shutil.copyfile(original,dest)
            assert sha(dest.read_bytes())==src["jpeg_sha256"]


def render(out,photos,browser):
    stats=json.loads((out/"summary.json").read_text())
    results={r["pair_id"]:r for r in rows(out/"pair_results.csv")}
    selected=[results[min(stats[k])] for k in ("max_cases","min_cases")]
    case_files(out,selected,photos)
    n=stats["pairs"]; npic=stats["photos"]
    a=stats["assignments"]; m=stats["M"]; mi=stats["mirror"]
    old=stats["old_both"]; grp=stats["same_choice"]["all_six_same"]; og=stats["same_choice"]["original_same"]
    prop=lambda k,d:f"{k}/{d}（{100*k/d:.2f}%）"
    answers=[
        ("方向分配未高于随机水平。",
         f'正确 {prop(a["correct"],n)}，颠倒 {prop(a["reversed"],n)}，同分 {prop(a["tie"],n)}。对非同分的 {stats["sign_test_n"]} 对做双侧精确符号检验，零假设为 50%，p = {stats["sign_test_two_sided_p"]:.3f}。'),
        ("同一张照片与自身镜像竞争，也很少两句都辨对。",
         f'{npic} 张中，两句都对 {mi["both_correct"]} 张、仅对一句 {mi["one_correct"]} 张、都错 {mi["both_wrong"]} 张、含同分 {mi["has_tie"]} 张。任一句同分即归入“含同分”，每张照片只计一次。'),
        ("原来始终选同一张图的案例分成两类。",
         f'三种旧策略对两句提示六次都选同一候选的 {sum(grp.values())} 对，方向分差分对 {grp["correct"]} 对、选反 {grp["reversed"]} 对。{selected[0]["pair_id"]} 分对，{selected[1]["pair_id"]} 选反。仅按原图两题选同图统计，则是 {og["correct"]} 对分对、{og["reversed"]} 对选反。')]
    title="两次评分能读出照片里的左右关系吗？"
    subtitle="HPS v2.1  ·  COCO 2017 validation  ·  64 对自然照片  ·  离线复算"
    headers=["策略","每图单题 / 两题去重评分次数","单题用相反提示","双题全对"]
    strategies=[
        ["原图","1 / 2","否",f'{old["original"]}/{n}'],
        ["只翻图","2 / 4","否",f'{old["image_only"]}/{n}'],
        ["翻图＋同步改词","2 / 4","是，询问镜像",f'{old["synchronized"]}/{n}'],
        ["原图方向分差","2 / 2","是，询问同一原图",f'{a["correct"]}/{n}']]
    method="左右真值由 COCO 的 X、Y 框恢复。每张原图取 e = 左提示分 − 右提示分，把 e 较大的照片分给左题。M = e(真实左图) − e(真实右图)。两题必须各用一张照片，因此换词时换图由规则保证；需要检验的是 M 的符号。"
    budget=f'单题固定两次评分，方向分差分对 {a["correct"]} 对，只翻图两题各自排序时全对 {old["image_only"]} 对。新规则还利用了一左一右的约束，两种决策的成功条件不同。'
    spread=f'M 分布（HPS 分）　最小 {m["min"]:+.6f}　Q1 {m["q1"]:+.6f}　中位 {m["median"]:+.6f}　Q3 {m["q3"]:+.6f}　最大 {m["max"]:+.6f}'
    md=["# "+title,subtitle,*["**"+lead+"** "+detail for lead,detail in answers],
        "[查看各口径下全部分对、选反与同分的图对 ID](same_choice_groups.csv)",
        md_table(headers,strategies),method,budget,spread]
    html_cards=[]
    for kind,r in zip(("最大","最小"),selected):
        case_title=f'{r["pair_id"]} · M {kind}，'+("分配正确" if r["assignment"]=="correct" else "分配颠倒")
        case_headers=[]; pics=[]; image_tags=[]; scores=[[],[]]
        for side in "AB":
            label=f'{side} · 真'+("左" if side==r["true_left"] else "右")
            img=Path("photos")/f'{int(r["image_"+side]):012}.jpg'
            case_headers.append(label)
            pics.append(f'![{label}]({img.as_posix()})')
            image_tags.append(f'<figure><figcaption>{label} · COCO {r["image_"+side]}</figcaption><img src="{(out/img).resolve().as_uri()}"></figure>')
            scores[0].append(r[side+"_left"])
            scores[1].append(r[side+"_right"])
        score_header=["原图评分",*case_headers]
        score_rows=[["左提示",*scores[0]],["右提示",*scores[1]]]
        direction_m=f'M = {float(r["M"]):+.13f}'.rstrip("0")
        md+=["### "+case_title,r["prompt_left"]+"<br>\n"+r["prompt_right"],
             md_table(case_headers,[pics]),md_table(score_header,score_rows),"**"+direction_m+"**"]
        html_cards.append('<article><h3>'+case_title+'</h3><div class="pics">'+"".join(image_tags)+'</div><div class="prompts">'+
                          escape(r["prompt_left"])+"<br>"+escape(r["prompt_right"])+"</div>"+html_table(score_header,score_rows)+
                          '<strong class="margin">'+direction_m+"</strong></article>")
    always=[r for r in results.values() if r["all_six_same"]=="1"]
    extra=[r for r in results.values() if r["original_same"]=="1" and r["all_six_same"]=="0"]
    def id_line(items, outcome):
        return " ".join(r["pair_id"][1:] for r in items if r["assignment"]==outcome)
    id_title="六次都选同一候选的图对（编号前缀 F）"
    id_rows=[["分对",id_line(always,"correct")],["选反",id_line(always,"reversed")]]
    extra_line=f'仅按原图两题选同图统计，还包括分对的 {id_line(extra,"correct")}、选反的 {id_line(extra,"reversed")}。'
    md+=["### "+id_title,md_table(["现在","编号"],id_rows),extra_line,
         "[逐对结果及完整精度](pair_results.csv) · [128 张照片的镜像判定](mirror_results.csv) · [图片来源与复算](README.md)"]
    (out/"report.md").write_bytes(("\n\n".join(md)+"\n").encode())
    if not browser:
        return
    css="""
    @page { size:A4; margin:13mm 14mm 12mm; }
    *{box-sizing:border-box} body{margin:0;color:#182f37;font:10.15pt/1.44 'Noto Sans SC','Microsoft YaHei',sans-serif}
    header{border-top:3px solid #1d665f;padding:3mm 0 2mm}h1{font-size:22pt;line-height:1.2;margin:0}
    .sub{font-size:9pt;color:#486269;margin:2mm 0 0}p{margin:1.8mm 0}b{color:#134b46}
    h2{font-size:13pt;margin:3mm 0 1.5mm}h3{margin:0 0 1mm;font-size:11pt;color:#164d48}
    table{border-collapse:collapse;width:100%;font-size:9.3pt;text-align:left;font-variant-numeric:tabular-nums}
    th{color:#214746;background:#e4eeec}td,th{padding:1mm 1.6mm;border-bottom:.4pt solid #c2cfcc}
    .note,.spread{font-size:9pt}.spread{padding:1.5mm 0;color:#114e48;font-weight:600}
    .cases{display:grid;grid-template-columns:1fr 1fr;gap:4mm} article{min-width:0;border-top:1pt solid #759792;padding-top:2mm}
    .pics{display:grid;grid-template-columns:1fr 1fr;gap:1mm}figure{margin:0}figcaption{font-size:8pt;color:#3a5155;margin-bottom:1mm}
    figure img{display:block;width:100%;height:27mm;object-fit:contain;background:#edf1ef}
    .prompts{font:9pt/1.32 Arial,sans-serif;margin:1.5mm 0}article td,article th{font-size:8.4pt;padding:1mm .8mm;white-space:nowrap} .margin{display:block;margin-top:1mm;font-size:9.5pt}
    .ids{font-size:8.7pt;margin-top:3mm}.ids p{margin:1mm 0}.ids strong{color:#164d48}
    footer{border-top:1px solid #c2cfcc;margin-top:2mm;padding-top:1.5mm;color:#3c545a;font-size:8pt}
    """
    html=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>'+css+'</style>'
          +'<header><h1>'+title+'</h1><p class="sub">'+subtitle+'</p></header>'
          +''.join('<p><b>'+lead+'</b> '+detail+'</p>' for lead,detail in answers)
          +'<h2>同一批图对的四种读法</h2>'+html_table(headers,strategies)
          +'<p class="note">'+method+'</p><p class="note">'+budget+'</p><p class="spread">'+spread+'</p>'
          +'<h2>方向分差的两个极端案例</h2><div class="cases">'+''.join(html_cards)+'</div>'
          +'<section class="ids"><strong>'+id_title+'</strong>'+''.join('<p>'+label+'　'+values+'</p>' for label,values in id_rows)+'<p>'+extra_line+'</p></section>'
          +'<footer>来源：冻结的 COCO 实例框与 512 条 HPS 缓存评分。每个案例的两张原始照片完整展示。逐对 ID、照片来源和完整分数见仓库 P2/direction_evidence。</footer></html>')
    with TemporaryDirectory() as temp:
        temp=Path(temp)
        page=temp/"report.html"; page.write_text(html,encoding="utf8")
        dest=temp/"report.pdf"
        subprocess.run([str(browser),"--headless","--disable-gpu","--no-pdf-header-footer","--allow-file-access-from-files",
                        "--user-data-dir="+str(temp/"profile"),"--print-to-pdf="+str(dest),page.as_uri()],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
        assert dest.is_file()
        shutil.copyfile(dest,out/"report.pdf")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=HERE)
    p.add_argument("--photos",type=Path,help="Cache of original COCO val2017 JPEGs (needed only on first export)")
    p.add_argument("--browser",type=Path,help="Chrome or Edge executable for printing the one-page PDF")
    args=p.parse_args()
    render(args.out.resolve(),args.photos,args.browser)
