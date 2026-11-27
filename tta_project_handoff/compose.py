"""Make the editable, self-contained print HTML from verified data and case PNGs."""
from pathlib import Path
import json,csv,hashlib,base64,html
R=Path(__file__).resolve().parent
a=json.loads((R/"audit.json").read_text(encoding="utf8"));cases=a["cases"]
pixel_records={r["filename"]:r for r in csv.DictReader((R/"case_inputs.csv").open(encoding="utf8"))}
def img(i,s,v):
    name=f"{i}-{s}-{v}.png";raw=(R/"assets"/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==pixel_records[name]["png_sha256"]
    data=base64.b64encode(raw).decode()
    return f'<img alt="{i}，候选 {s}，{v}" src="data:image/png;base64,{data}">'
def signed(x):return f"{x:+.6f}".replace("-","−") if x else "0.000000"
def sline(i,version):
    c=cases[i];z=c["combo"][version]
    return f'原图 A−B <b>{signed(c["d0"])}</b> → 三分均值 <b>{signed(z[2])}</b>'
def label(v):
    return {"O":"原图","B090":"亮度 ×0.90","Tdown":"下移 3%","Tleft":"左移 3%","Tright":"右移 3%","Tup":"上移 3%"}[v]
def grid(i,vs):
    hd='<span></span>'+''.join(f'<span>{label(v)}</span>' for v in vs)
    cells=''.join(f'<b class="side">{side}</b>'+''.join(img(i,side,v) for v in vs) for side in "AB")
    return f'<div class="pics" style="--cols:{len(vs)}">{hd}{cells}</div>'
def tab(head,body,cl=""):
    return f'<table class="{cl}"><thead><tr>'+''.join(f'<th>{x}</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in r)+'</tr>' for r in body)+'</tbody></table>'
def src(txt):return f'<div class="source">资料　{txt}</div>'
def num(x):return f'<strong class="num">{x}</strong>'
def page(idx,section,heading,body):
    return f'<section class="page p{idx}"><div class="folio"><span>测试时增强与偏好排序</span><span>{section}</span></div><main><h1>{heading}</h1>{body}</main><footer><span>ImageRewardDB validation　·　HPS v2.1</span><b>{idx:02} / 08</b></footer></section>'
def subtitle(t):return f'<h2>{t}</h2>'
def matrix(i):
    c=cases[i];rows=[]
    for j,f in enumerate(["0.90","0.95","1.05","1.10"]):
        line=[f];h=c["human"]
        for p in c["matrix"][4*j:4*j+4]:
            outcome="选对" if p==h else "同分" if p=="同分" else "选错"
            cl={"选对":"yes","选错":"no","同分":"tie"}[outcome]
            line.append(f'<span class="state {cl}">{p} · {outcome}</span>')
        rows.append(line)
    return tab(["亮度系数","左移","右移","上移","下移"],rows,"matrix")
def scoretable(i,versions):
    c=cases[i];rr=[]
    for title,v in versions:
        if v=="O":x,y=c["orig"];d=c["d0"];w="A" if d>0 else "B"
        else:x,y,d,w=c["combo"][v]
        state="选对" if w==c["human"] else "同分" if w=="同分" else "选错"
        rr.append([title,f"{x:.6f}",f"{y:.6f}",signed(d),w+" · "+state])
    return tab(["比较条件","A 分数","B 分数","A−B","模型选择"],rr,"compact")
def figure(n,title,content,caption,source):
    return f'<div class="figure"><div class="figtitle"><b>图 {n}</b>{title}</div>{content}<div class="caption">{caption}</div>{src(source)}</div>'
def bar(val,maxval,cl,txt):
    return f'<div class="barrow"><span>{txt}</span><div class="track"><i class="{cl}" style="width:{100*val/maxval:.3f}%"></i></div><strong>{val:.2f}</strong></div>'
def what(txt):return f'<p class="lead">{txt}</p>'
pages=[]
# Opening example uses exactly the same A/B pixels and scores as the discovery case.
c=cases["003"]; vv=c["single"]
pairimg=''.join(f'<div>{img("003",s,"O")}<b>候选 {s}</b></div>' for s in "AB")
p1='''<div class="deck">阶段发现与实验交接</div>'''+what("给同一张图换个亮度或位置，评分器还会选同一个赢家吗？我们用已有人工排名作参照，检验多看两个版本值不值得。")+f'<div class="opening"><div class="opening-img">{pairimg}</div><div><h2>先读懂一对图</h2><p>旧样本 003 要求一张高反差、带颗粒的黑白人像。HPS 是给「图片＋提示词」打分的偏好模型。<b>候选分差 = A 分 − B 分</b>，正数选 A，负数选 B，零为同分。</p></div></div>'
p1+=tab(["003 · 原提示词","候选 A","候选 B","A−B","选择"],[[tag,*[f"{x:.6f}" for x in vv[v]],signed(vv[v][0]-vv[v][1]),"A"] for tag,v in [("原图","O"),("两图都调亮 10%","B110")]],"compact")
p1+='<p class="caption">两图都涨分，B 涨得更多，A 的领先缩小但仍获选。人工更喜欢 B。</p>'+src("P1/baseline/scores.csv、pairs.csv；003 原图取自 P3/old_cases。")
p1+='<p>测试时增强（TTA）是在模型权重固定时，把同一张图的不同版本分别评分，再把分数合起来选赢家。我们用 ImageRewardDB 验证集中人对<b>原图</b>的排名作参考，同时追踪纠正和改错。</p>'
p1+=subtitle("目前能交给下一位队友的三个答案")+'''<div class="take"><b>夜景要求</b><span>它改变了单图对调亮的评分响应。同一对候选常往同一方向变，共同变化在 A 减 B 时抵消。</span></div><div class="take"><b>平移的得失</b><span>在 256 对新图中，平移补到 7 个光度策略漏掉的错例；均匀选参数时，新增改错超过新增纠正。</span></div><div class="take"><b>改判的条件</b><span>原始领先很小时才容易越过零；同一图对能否越过，还取决于画面向哪个方向平移。</span></div>'''
# Let a new reader see the three answers immediately, then work through the real pair.
intro,example=p1.split('<div class="opening">',1)
example,answers=example.split(subtitle("目前能交给下一位队友的三个答案"),1)
p1=intro+answers+'<div class="opening">'+example
pages.append(page(1,"项目问题与答案","测试时增强如何改变<br>图像偏好排序",p1))
p2=what("实验 P1 研究亮度与提示词的交互，P2 比较增强组合，两者共用 64 对图。P3 换用 256 个新提示词复验平移。实验编号 P1 与下文的夜景提示词代号 P1 分属两套编号。")
p2+=tab(["原先的疑问","实验改动","新增证据"],[
["<b>P1 基线 · 只调亮会换赢家吗？</b>","同一图对交叉使用原提示词 P0、追加夜景的 P1，以及原图、亮度 ×1.10。","P0 下单看调亮有 2 次严格翻转；与原图等权平均后严格翻转为 0。"],
["<b>P1 短提示词 · 夜景要求进模型了吗？</b>","审查 tokenizer；去掉明确光照描述，缩写成 Q0，追加完整夜景短句得 Q1。","23 条 P0 自带光照；9 条 P1 截断，其中 3 对模型实际收到的词元完全相同。"],
["<b>P1 分解 · 分数动了，赢家为何少变？</b>","固定 Q0/Q1 评分，把两张图各自对调亮的反应拆成共同与差异分量。","128 张图里 78 张的调亮响应变得更负；差值只保留两图变化不同的部分。"],
["<b>P1 重配 · 同向只是总体趋势吗？</b>","固定每张图的响应，改和其他提示词下的候选重配。","真实配对 42 对同向；跨提示词重配期望 27.67 对。"],
["<b>P2 · 多看哪两种图？</b>","回到 P0，比较 B 亮度、C 对比度、J JPEG 压缩、T 平移。同类或跨类均用三次评分。","003 出现光度策略漏掉的纠正；008 则被 BT 的所有搭配改错。"],
["<b>P3 · 换提示词还成立吗？</b>","排除旧 ID、重复文本及已用图片，冻结同一验证集的 256 对；主比 BT 与 BC。","小分差组重现平移的纠正和改错；主比较的准确率差区间跨零。"]],"stages")
p2+=src("自上而下对应 P1/baseline、P1/prompts、P1/decomposition、P1/pairing、P2、P3；逐表入口见第 8 页。")
p2+=subtitle("005 与 053：等权平均怎样压回一次翻转")
p2+=tab(["旧样本 · P0","人工赢家","原图分差","单看调亮 10%","原图＋调亮均值"],[[i,cases[i]["human"],signed(cases[i]["pilot"][0]),signed(cases[i]["pilot"][1]),signed(cases[i]["pilot"][2])] for i in ["005","053"]],"compact")
p2+='<p class="caption">005 的有害翻转被压到同分；053 的有益翻转被压回原选择 A。这里展示单次调亮与两次均值；另一个实验中 053 的 BC 使用三次评分。</p>'+src("P1/baseline/scores.csv、pair_results.csv。")
p2+='<div class="note"><b>人工标签的适用范围</b>　P0 是数据集原文，人的排名对应原文和原图。P1、Q0、Q1 改过任务，只能谈模型响应与改选。旧样本的多轮结果共享同一批图。</div>'
pages.append(page(2,"研究设计","三个实验如何衔接",p2))
# Figure 1 displays exact analytic means and empirical quantile references.
def rangeline(row,scale,unit):
    low,high,mean,obs=[float(row[k]) for k in ("null_q025","null_q975","exact_null_mean","observed")]
    scalevalue=100 if unit=="%" else 1
    fmt=lambda v:f"{v*scalevalue:.2f}".rstrip("0").rstrip(".")+unit
    def pct(v):return 100*v/scale
    return f'<div class="rangehead"><b>跨提示词 {fmt(mean)}</b><strong>真实配对 {fmt(obs)}</strong></div><div class="ruler"><div class="strip" style="left:{pct(low)}%;width:{pct(high-low)}%"></div><i class="dotgray" style="left:{pct(mean)}%"></i><i class="dotgreen" style="left:{pct(obs)}%"></i></div><div class="rangelabel"><span>0</span><span>重配中央 95%　{fmt(low)} 至 {fmt(high)}</span><span>{fmt(scale)}</span></div>'
f1=figure(1,"同一批单图响应，改变配对后会怎样？",'<div class="rangebox"><h4>两图同向且均非零的图对数</h4>'+rangeline(a["random_reference"]["same_nonzero"],64," 对")+'<h4>共同分量占交互变化平方和的比例</h4>'+rangeline(a["random_reference"]["R"],1,"%")+'</div>',"64 对。灰带为 100 万次跨提示词重配的中央 95% 范围，灰点为精确重配期望，绿点为真实候选。两个指标的真实值都落在灰带右侧。","P1/pairing/summary.csv、null_distribution.csv；单图响应取自 P1/decomposition/pair_decomposition.csv。")
p3=what("先看单张图，再看两张图是否一起变。追加夜景要求后，调亮引起的分数变化在很多图片上都变得更负。真实候选之间还有额外的同步性。")
p3+=subtitle("对一张图，量两次“调亮带来的分数变化”")
p3+='<p class="caption">Q1 在 Q0 后完整追加同一句英文：The scene is at night, with dim lighting.</p>'
p3+='<p>先在短基础提示词 Q0 下算「调亮分 − 原图分」，再在追加夜景、低照度要求的 Q1 下算一次。后者减前者记作 <b>h</b>。h 为负，既可能是调亮加分变少，也可能是调亮扣分变多。128 张图里，78 张为负、34 张为正、16 张为零。</p>'
p3+=f1
p3+=subtitle("为什么共同变化能很大，分差却动得很少？")
p3+='<p>如果两图的 h 都是同一个数，在 A−B 时会完全抵消。把两者的平均记作共同分量，把两者之差的一半记作差异分量。各自平方后在 64 对上求和，<b>83.24%</b> 落在共同分量上。这个百分比描述变化量的分配；若直接数图对，同向且非零的是 <b>42/64</b>。</p>'
p3+='<div class="note"><b>能说到哪一步</b>　重配时每张图仍保留在自己提示词下算出的 h，只改变谁与谁配对。该对照保留了许多图片整体偏负的趋势。真实候选同时共享文本、也常有相似内容，两者各自的作用仍待区分。</div>'
p3+='<p>单图 h 的绝对值中位数仅为 <b>0.000488</b> 分。只看确实朝平局移动的图对，调亮 10% 通常只走过原先领先幅度的一小段：Q0 的逐对覆盖比例中位数为 9.82%，Q1 为 8.94%。</p>'+src("P1/prompts/scores.csv；P1/decomposition/pair_decomposition.csv、summary.csv。")
pages.append(page(3,"发现 A · 旧 64 对","同一对候选更容易一起变",p3))
p4=what("每张图总共评三次。BT 用原图、一个亮度版本、一个平移版本；BC 保留相同的原图和亮度档，把第三次换成一个对比度版本。")
p4+='<p>每类 4 个参数版本，跨类有 4 × 4 = 16 套。每套中 A、B 用相同参数，各自取三个分数的均值后再选赢家。<b>4/16</b> 表示 16 套中有 4 套触发该结果。同类从 4 个版本中选两个不同版本，共 6 套。</p>'
budget=a["budget"]
fb='<div class="bars">'+''.join(f'<div><h4>{name}</h4>'+''.join(bar(budget[st][k],4,cl,st) for st,cl in [("BT","green"),("BC","gray")])+'</div>' for k,name in [("wrong_right","原图选错 → 纠正"),("right_wrong","原图选对 → 改错")])+'</div>'
p4+=figure(2,"均匀选参数时，纠正和改错各有多少",fb,"256 对。小数是对参数搭配平均得到的期望对数。BT 比 BC 多纠正 1.25 对，也多改错 1.75 对。图中柱长按未舍入值绘制。","P3/transitions.csv、scores.csv。")
p4+='<p><b>7 对</b>原始错例在 BB、CC、BC 的全部搭配下都没选对，BT 至少有一套能纠正；TT 也能纠正这 7 对。7 衡量能覆盖几个病例，1.25 是计入全部 256 对、均匀选搭配后的平均新增纠正。</p>'
p4+=tab(["策略","期望选对","期望选错","期望同分","严格选对率"],[["BT","174.9375","81.0000","0.0625","68.335%"],["BC","175.3125","80.5000","0.1875","68.481%"]],"compact")
p4+='<p class="caption">BT−BC 为 −0.146 个百分点；按整对重采样 10,000 次的 95% 区间为 [−1.465, +1.270] 个百分点。区间允许两个方向的差异，当前无法确定总体优劣。</p>'+src("P3/strategy_summary.csv、endpoint.json；逐对贡献由原始评分重算。")
blocks=''.join(f'<div><span>第{j}组 · 64 对</span><strong>{r["any_BT_BC_change"]}</strong><small>BT {r["BT_changed_pairs"]}　BC {r["BC_changed_pairs"]}</small></div>' for j,r in enumerate(a["margin_groups"][:4],1))
p4+=figure(3,"按原始领先幅度从小到大，哪些组至少一次改判？",f'<div class="quart">{blocks}</div>',"256 对。每组大数为 BT 或 BC 改判的不同图对数，重合只算一次。全部 21 对落在第一组，其原始领先幅度至多 0.008301 分；另外三组均为零。","P3/margin_groups.csv、pair_contributions.csv。")
p4+='<p>在第一组，平移导致的绝对分差位移中位数为 0.004639，对比度为 0.001221。三分均值要换赢家，两个增强造成的分差位移之和必须逆着原领先方向，且超过原领先幅度的三倍。平移更容易移动分差，移动的方向决定它带来纠正还是改错。</p>'
pages.append(page(4,"发现 B / C · 新 256 对","平移补到错例，也会损失对例",p4))
# Discovery cases share one page so both consequences can be inspected together.
def oldcase(i,heading,v,text,choice_desc,lesson):
    return f'<article class="oldcase"><h2><span>{i}</span>　{heading}</h2><div class="oldbody">{grid(i,["O","B090",v])}<div class="oldinfo"><h4>原提示词摘要</h4><p>{text}</p><h4>人工参考与模型选择</h4><p>{choice_desc}</p><h4>所示 BT 搭配</h4><p>原图、亮度 ×0.90、{label(v)}，三个分数等权平均。</p></div></div><p class="lesson">{sline(i,"B090 "+v)}。{lesson}</p></article>'
p5=what("以下两对都使用数据集原提示词 P0。每个网格第一行为 A、第二行为 B；三列是实际独立评分的输入。第三列的方向指画面移动方向。")
p5+=oldcase("003","只有下移让 BT 选回人工赢家","Tdown","大画幅黑白男士肖像，白色背景、聚光灯、高反差，有景深和颗粒感。","人工选 <b>B</b>；原图选 A，所示 BT 选 <b>B</b>。","下移四档均纠正，BT 共 4/16，TT 也有 4/6。")
p5+=oldcase("008","四个方向都把原选择改错","Tleft","森林里一只长着猫头、身上略有绒毛的恐龙。","人工选 <b>A</b>；原图选 A，所示 BT 选 <b>B</b>。","左移是其中一例，四方向 × 四档亮度全部改错（16/16）。")
p5+='<p class="caption">案例图均为模型的 224 × 224 实际输入。平移按边长 3% 取整到像素、反射填边；官方预处理按长边双三次缩放、补边、归一化，展示时再反归一化。</p>'+src("P1/baseline/pairs.csv 固定 A/B 和人工赢家；P2/scores.csv、combination_results.csv 提供分数。原图与平移图复用 P3/old_cases/assets；亮度图用同一预处理重建。")
pages.append(page(5,"发现案例 · 旧 64 对","同一种平移策略的两面",p5))
p6=what("原提示词要求一位开心的胖男士，穿夏威夷衬衫和轮滑鞋。人工选 A，原图模型选 B。")+'<p class="caption">原文　'+html.escape(cases["N032"]["prompt"])+'</p>'
p6+=grid("N032",["O","B090","Tup","Tdown"])
p6+='<p class="caption">所示两套 BT 都取原图与亮度 ×0.90；第三次单独评分分别取上移或下移图。上、下方向各列只移动 3%。</p>'
p6+=scoretable("N032",[("原图","O"),("BT · 上移","B090 Tup"),("BT · 下移","B090 Tdown")])
p6+=subtitle("换亮度档，上移的结果还在吗？")+matrix("N032")
p6+='<p class="lesson">只有上移能纠正，而且四档亮度全部成立；其余三个方向的 12 套搭配都保留了原来的错误选择 B。</p>'
p6+=src("P3/pairs_frozen.csv、scores.csv、combination_results.csv、direction_counts.csv。图像取自 validation_2 原始 WebP，用 P2/views.py 和评分时的官方预处理重建。")
pages.append(page(6,"复验案例 · 新 256 对","N032　只有上移能纠正",p6))
p7=what("提示词描述被扔在泥土里的拍立得，原文用词为 polaroid。人工选 B，原图模型也选 B。这一对的原始领先幅度也很小，多方向重复出现的是改错。")+'<p class="caption">原文（保留拼写）　'+html.escape(cases["N141"]["prompt"])+'</p>'
p7+=grid("N141",["O","B090","Tleft","Tright"])
p7+='<p class="caption">左移和右移是两个独立的第三次输入。这里固定亮度 ×0.90，以便直接看向相反方向移动时分差是否回到零的另一侧。</p>'
p7+=scoretable("N141",[("原图","O"),("BT · 左移","B090 Tleft"),("BT · 右移","B090 Tright"),("BC · 对比度 ×0.90","B090 C090")])
p7+=subtitle("同一对图上，多少亮度档会改错？")
p7+=tab(["平移方向","左移 3%","右移 3%","上移 3%","下移 3%"],[["改错 / 四档","4 / 4","4 / 4","4 / 4","4 / 4"]],"compact")
p7+='<p class="lesson">BT 在 16/16 套搭配下都把 B 改成 A；BC 在相同预算下仍有 9/16 套保留人工赢家 B。</p>'
p7+='<p><b>回到全部 256 对。</b>17 对至少一次被 BT 改判，其中 16 对能在同一方向跨亮度档复现，9 对会被多个方向触发。这些重复观察支持参数方向确实有影响；它们仍然来自同一个图对，估计区间时每对只算一个样本。</p>'
p7+=src("P3/pairs_frozen.csv、scores.csv、combination_results.csv、direction_counts.csv。图像取自 validation_1 原始 WebP，重建脚本及逐视图哈希在交接目录 case_inputs.py、case_inputs.csv。")
pages.append(page(7,"复验案例 · 新 256 对","N141　每个方向都改错",p7))
p8=what("想核对一个结论，先找到对应批次的 CSV，再决定是否需要重新评分。以下命令均在服务器上进入表中所列相对目录后执行。")
p8+='<p class="cmdp">先在项目根目录执行 <code>HPS="$PWD/P1/baseline/.venv/bin/python"</code>，CPU 环境取 <code>CPU=/usr/bin/python3</code>。表中分析命令只需已有缓存。</p>'
p8+=tab(["想复查什么","相对目录与优先检查的文件","复跑命令（目录内）"],[
["原图分差与<br>等权均值",'<code>P1/baseline/</code><br>scores.csv → pair_results.csv','<code>$HPS -B analyze.py</code>'],
["截断与短提示词",'<code>P1/prompts/</code><br>old_prompt_audit.csv、scores.csv','<code>$HPS -B analyze.py</code>'],
["共同与差异分量",'<code>P1/decomposition/</code><br>pair_decomposition.csv','<code>$CPU -B analyze.py</code>'],
["随机重配参照",'<code>P1/pairing/</code><br>summary.csv、null_distribution.csv','<code>$CPU -B analyze.py</code><br><code>$CPU -B present.py</code>'],
["旧 64 对的互补性",'<code>P2/</code><br>scores.csv、combination_results.csv','<code>$CPU -B analyze_p2.py</code>'],
["新 256 对的增益<br>与方向",'<code>P3/</code><br>transitions.csv、direction_counts.csv','<code>$CPU -B analyze.py</code><br><code>$CPU -B verify.py</code>']],"routes")
p8+='<p class="caption">若确需从原图重新评分，模型推理入口例如在 P3/ 执行 <code>$HPS -Bu score_p3.py --limit 2</code> 再执行 <code>$HPS -Bu score_p3.py --limit 256</code>。已有匹配检查点会直接命中；从头推理请按该轮 README 在新副本开始。</p>'
p8+=subtitle("哪些预期还没有得到支持")
p8+='<p>短提示词下，夜景要求使两图亮度响应的差异幅度在 23 对上增强、34 对上减弱、7 对不变。旧 64 对上，BC 较原图增加的 0.0625 个期望正确图对，全部来自 053 的 1/16 套搭配。新 256 对的主比较区间也跨零，混合增强能否稳定提高整体准确率仍待检验。</p>'
p8+=subtitle("下一步优先验证")
p8+='<div class="next"><b>1</b><span><strong>按原始分差分配预算。</strong>在独立样本预先固定触发阈值，小分差才追加评分；与相同平均预算的固定策略比较纠正、改错、同分与成本。现有实验尚未测试这种分配策略。</span></div><div class="next"><b>2</b><span><strong>拆开共享文本和相似画面的作用。</strong>控制同一图像内容，交叉更换可比提示词；再在相同提示词下比较内容相似度不同的图对，检验哪部分同步性能够保留。</span></div>'
p8+=src("命令取自各阶段 README；证据边界来自 P1/prompts/summary.csv、P2/combination_results.csv、P3/endpoint.json。")
pages.append(page(8,"复查入口与研究交接","下一位同学从哪里接手",p8))
css=(R/"print.css").read_text(encoding="utf8")
(R/"TTA阶段交接报告.html").write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>测试时增强如何改变图像偏好排序：阶段发现与实验交接</title><style>'+css+'</style></head><body>'+''.join(pages)+'</body></html>',encoding="utf8")

