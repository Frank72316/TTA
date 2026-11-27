"""Read-only P2 audit and rendering through HPS's actual validation transform."""
import html
import re
from collections import Counter
from zipfile import ZipFile
import numpy as np
from PIL import Image
from common import *

def main():
    from hpsv2.src.open_clip.transform import image_transform
    env=file_json(P2/"environment.json")
    assert digest(P2/"views.py")==env["source_hashes"]["views.py"]
    tf=image_transform(224,False,resize_longest_max=True)
    assert re.sub(r" at 0x[0-9a-fA-F]+","",repr(tf))==env["preprocess"]
    scores=read_csv(P2/"scores.csv")
    ps=read_csv(PILOT/"pairs.csv")
    bank={(r["pair_id"],r["side"],r["view"]):r for r in scores}
    assert len(bank)==len(scores)==2176
    found={}
    for p in ps:
        i=p["pair_id"]; human=p["human_winner_P0"]
        d={v:float(bank[i,"A",v]["score"])-float(bank[i,"B",v]["score"]) for v in ["O",*sum(FAMILIES.values(),[])]}
        events={}
        for s in ["BB","CC","JJ","TT","BC","BJ","BT","CJ","CT","JT"]:
            events[s]=[(x,y,(d["O"]+d[x]+d[y])/3) for x,y in combinations(s)]
        if any(choice(z)!=choice(d["O"]) for vals in events.values() for _,__,z in vals):
            found[i]=(p,d,events)
    assert set(found)=={"003","005","008","031","053"}
    for i,s,state,n in [("003","TT","correct",4),("003","BT","correct",4),("008","BT","wrong",16)]:
        p,d,ev=found[i]
        assert sum(outcome(choice(z),p["human_winner_P0"])==state for _,__,z in ev[s])==n
    folder=ROOT/"old_cases";assets=folder/"assets";assets.mkdir(exist_ok=True,parents=True)
    order=["003","008","053","005","031"]
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>P3 平移案例</title><style>body{max-width:940px;margin:32px auto;padding:0 18px;font:16px/1.55 system-ui;color:#173247}section{margin:40px 0 64px}table{border-collapse:collapse;margin:12px 0 24px}td,th{border:1px solid #c3cdd2;padding:6px 10px;text-align:center}img{max-width:100%;height:auto;vertical-align:middle}.correct{background:#d9ece5}.wrong{background:#f4dcce}.tie{background:#ececec}a{color:#176865}code{font-size:1em}p{overflow-wrap:anywhere}</style><h1>平移如何改变旧案例的选择</h1><p>旧 64 对中，至少一个原三次策略改变选择的全部 5 对。下表分差均为 A 减 B；绿为选对，橙为选错，灰为同分。方向指画面移动方向。</p><nav>'+" · ".join(f'<a href="#c{i}">{i}</a>' for i in order)+'</nav>']
    details=[]
    for i in order:
        p,d,ev=found[i];human=p["human_winner_P0"]
        parts.extend([f'<section id="c{i}"><h2>图对 {i}</h2><p lang="en">{html.escape(p["P0"])}</p><p>人工赢家 <b>{human}</b>，原图分差 {d["O"]:+.10f}，原图选择 {choice(d["O"])}。BT 选对 {sum(choice(z)==human for _,__,z in ev["BT"])}/16；TT 选对 {sum(choice(z)==human for _,__,z in ev["TT"])}/6。</p>'])
        for s in ("A","B"):
            src=Path(p["image_"+s]);fname=f"{i}-{s}.webp"
            with ZipFile(PILOT/"data"/(src.parent.name+".zip")) as z:(assets/fname).write_bytes(z.read(src.name))
            parts.append(f'<a href="assets/{fname}">打开 {s} 原始 WebP</a>　')
        parts.append('<h3>模型输入 · 224 × 224</h3><table><tr><th>视图</th><th>A</th><th>B</th></tr>')
        rgbs={s:original(p,s) for s in ("A","B")}
        for v,label in zip(["O",*FAMILIES["T"]],["原图","左移","右移","上移","下移"]):
            parts.append(f"<tr><th>{label}</th>")
            for s,rgb in rgbs.items():
                im,_=make_view(rgb,v)
                assert hashlib.sha256(im.tobytes()).hexdigest()==bank[i,s,v]["pixel_sha256"]
                final=final_rgb(tf,im)
                base=final_rgb(tf,rgb)
                assert final.shape==(224,224,3)
                assert np.isclose(np.abs(final-base).mean(dtype=np.float64)*255,float(bank[i,s,v]["input_mean_abs_RGB255"]),atol=1e-7)
                file=f"{i}-{s}-{v}.png";Image.fromarray(np.uint8(np.rint(255*np.clip(final,0,1)))).save(assets/file)
                parts.append(f'<td><img src="assets/{file}" width="224" height="224" alt="{s} {label}"></td>')
            parts.append('</tr>')
        parts.append('</table><h3>BT 三次评分 · 行为亮度，列为平移</h3><table><tr><th>系数</th><th>左</th><th>右</th><th>上</th><th>下</th></tr>')
        def cell(x):
            result=outcome(choice(x),human);cn={"correct":"选对","wrong":"选错","tie":"同分"}[result]
            return f'<td class="{result}">{x:+.9f}<br>{cn} · {choice(x)}</td>'
        for b in FAMILIES["B"]:
            parts.append(f'<tr><th>{int(b[1:])/100:.2f}</th>')
            for bv,t,x in ev["BT"]:
                if bv==b:
                    parts.append(cell(x));details.append(dict(pair_id=i,method="BT",brightness=b,direction=t,d0=d["O"],margin=x,pick=choice(x),human=human))
            parts.append('</tr>')
        parts.append('</table><h3>仅原图与单个平移等权平均</h3><table><tr><th>左</th><th>右</th><th>上</th><th>下</th></tr><tr>')
        for t in FAMILIES["T"]:
            x=(d["O"]+d[t])/2
            parts.append(cell(x));details.append(dict(pair_id=i,method="O+T",brightness="—",direction=t,d0=d["O"],margin=x,pick=choice(x),human=human))
        parts.append('</tr></table></section>')
    (folder/"index.html").write_text("\n".join(parts)+"</html>\n",encoding="utf8")
    write_csv(ROOT/"old_case_checks.csv",details)
    print("Old 64: five changed pairs:",order,"003 TT 4/6 BT 4/16 corrections; 008 BT 16/16 harms; 50 model inputs verified.")
if __name__=="__main__": main()

