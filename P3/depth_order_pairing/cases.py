"""Draw the frozen examples on the upstream common RGB field with their measured values."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image,ImageDraw,ImageFont
from order_data import HERE,SOURCE,sha,read

def render(out,archive,font_file):
    assert sha(archive)==json.loads((SOURCE/"environment.json").read_text())["archive_sha256"]
    frames={r["frame_id"]:r for r in read(SOURCE/"frame_results.csv")}
    windows={(r["frame_id"],int(r["window"])):r for r in read(out/"windows.csv")}
    pairs={(r["frame_id"],int(r["A"]),int(r["B"])):r for r in read(out/"region_pairs.csv")}
    choices=[c for c in json.loads((out/"validation.json").read_text())["cases"] if c["first"]]
    if not choices:return
    def font(sz):return ImageFont.truetype(str(font_file),sz,index=2)
    sheet=Image.new("RGB",(1060,len(choices)*444),"#ffffff");d=ImageDraw.Draw(sheet)
    with ZipFile(archive) as z:
        for i,case in enumerate(choices):
            top=i*444;c=case["first"];fid=c["frame_id"];a,b=c["A"],c["B"];r=pairs[fid,a,b]
            title="距离差变准，远近判定维持" if case["kind"]=="unchanged" else "距离差变准，远近由错转对"
            d.rectangle((0,top,1060,top+46),fill="#203746")
            d.text((15,top+5),title+"     "+fid+"  ·  "+frames[fid]["scene"],font=font(23),fill="white")
            with Image.open(BytesIO(z.read(frames[fid]["frame"]))) as im:
                rgb=im.convert("RGB").crop((26,0,614,480))
            dr=ImageDraw.Draw(rgb)
            for win,tag,col in ((a,"A","#ffca50"),(b,"B","#60e4f5")):
                w=windows[fid,win];x,y=int(w["x0"]),int(w["y0"])
                dr.rectangle((x-2,y-2,x+33,y+33),outline="black",width=2)
                dr.rectangle((x,y,x+31,y+31),outline=col,width=3)
                dr.rectangle((x-1,y-30,x+54,y-2),fill="#203746")
                dr.text((x+3,y-32),f"{tag}{win:02}",font=font(20),fill=col)
            sheet.paste(rgb.resize((464,379),Image.Resampling.LANCZOS),(12,top+54))
            x=504
            d.text((x,top+52),f"真值均深差 A − B    {float(r['d']):+.5f} m",font=font(24),fill="#133d48")
            d.text((x,top+94),"配方",font=font(19),fill="#45545c")
            d.text((682,top+94),"预测 A − B (m)",font=font(19),fill="#45545c")
            d.text((884,top+94),"更远区域",font=font(19),fill="#45545c")
            for j,k in enumerate(("O","LL","LR","RR","RL")):
                y=top+135+42*j;q=float(r["q_"+k])
                if k in (c["before"],c["after"]):d.rectangle((x-4,y-2,1040,y+34),fill="#edf3f4")
                d.text((x,y),{"O":"O","LL":"O + L2 + L4","LR":"O + L2 + R4","RR":"O + R2 + R4","RL":"O + R2 + L4"}[k],font=font(19),fill="#233542")
                d.text((684,y),f"{q:+.5f}",font=font(20),fill="#172b3b")
                answer="A" if q>0 else "B" if q<0 else "同分"
                status={"correct":"对","wrong":"错","tie":"平"}[r["state_"+k]]
                d.text((890,y),answer+"（"+status+"）",font=font(19),fill="#172b3b")
            u,v=c["before"],c["after"]
            d.text((x,top+364),f"{u} → {v}    平方误差 (m²)",font=font(19),fill="#284c56")
            d.text((x,top+397),f"{float(r['sqerr_'+u]):.6f}  →  {float(r['sqerr_'+v]):.6f}",font=font(21),fill="#172b3b")
    sheet.save(out/"cases.jpg",quality=94,subsampling=0)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=HERE);p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    a=p.parse_args();render(a.out,a.archive,a.font)
