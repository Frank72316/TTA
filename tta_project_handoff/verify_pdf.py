"""Render a detached copy with Poppler, verify embedded fonts and the identity of each displayed candidate."""
import hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageChops,ImageStat
import numpy as np
from pypdf.generic import ContentStream
R=Path(__file__).resolve().parent
pdf=R/"TTA阶段交接报告.pdf";qa=R/"qa";qa.mkdir(exist_ok=True)
cases={"003":["O","B090","Tdown"],"008":["O","B090","Tleft"],"N032":["O","B090","Tup","Tdown"],"N141":["O","B090","Tleft","Tright"]}
def ids(i):return [f"{i}-{s}-{v}.png" for s in "AB" for v in cases[i]]
expected=[["003-A-O.png","003-B-O.png"],[],[],[],ids("003")+ids("008"),ids("N032"),ids("N141"),[]]
terms=["候选分差","已完成的实验对应哪些假设","同一对候选更容易一起变","平移补到错例","同一种平移策略的两面","只有上移能纠正","每个方向都改错","下一位同学从哪里接手"]
render=shutil.which("pdftoppm")
if not render:raise RuntimeError("pdftoppm required on PATH")
with tempfile.TemporaryDirectory(prefix="tta_handoff_detached_") as d:
    detached=Path(d)/"handoff.pdf";shutil.copyfile(pdf,detached)
    assert len(list(Path(d).iterdir()))==1
    subprocess.run([render,"-r","130","-png",str(detached),str(qa/"page")],check=True,cwd=d)
    reader=PdfReader(detached);assert len(reader.pages)==8
    details=[];fonts=set()
    for index,p in enumerate(reader.pages):
        text=p.extract_text();assert terms[index] in re.sub(r"\s","",text),(index,text[:100])
        res=p["/Resources"]
        for name,font in res["/Font"].items():
            f=font.get_object();fonts.add(str(f["/BaseFont"]))
            fontset=f.get("/DescendantFonts",[f])
            for face in fontset:
                descriptor=face.get_object()["/FontDescriptor"]
                assert any(k in descriptor for k in ("/FontFile","/FontFile2","/FontFile3")),f
        assert not p.get("/Annots")
        # Traverse actual painting operations so order (A first, B second) is also tested.
        seq=[str(args[0])[1:] for args,op in p.get_contents().operations if op==b"Do"]
        assert len(seq)==len(expected[index]),(index,seq)
        images={item.name.rsplit(".",1)[0]:item.image.convert("RGB") for item in p.images}
        errs=[]
        for key,filename in zip(seq,expected[index]):
            orig=Image.open(R/"assets"/filename).convert("RGB");seen=images[key]
            assert seen.size==orig.size==(224,224)
            err=max(ImageStat.Stat(ImageChops.difference(orig,seen)).rms);errs.append(err)
            assert err<4,(filename,err)
        details.append(dict(page=index+1,case_images=expected[index],rms_max=max(errs,default=0),embedded_fonts_verified=True))
assert len(list(qa.glob("page-*.png")))==8
(qa/"verification.json").write_text(json.dumps(dict(pages=8,detached_pdf_rendered=True,renderer="Poppler 130 dpi",fonts=sorted(fonts),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),details=details),indent=2,ensure_ascii=False),encoding="utf8")
for k in (0,1):
    im=Image.new("RGB",(2*500,2*707),"white")
    for j in range(4):
        p=Image.open(qa/f"page-{1+k*4+j}.png").convert("RGB");p.thumbnail((500,707));im.paste(p,(j%2*500,j//2*707))
    im.save(qa/f"overview-{k+1}.jpg",quality=88)
print("Checked 8 detached pages, 30 case image placements, embedded fonts:",fonts)

