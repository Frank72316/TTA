# TTA

对于你的研究——利用多个视角的2D奖励分布，估计3D资产的整体奖励——我建议采用 Test\-Time Augmentation（TTA，测试时增强）\+ Monte Carlo Sampling 的方法。

基本思路是：固定3D资产及其一个相机视角，对该视角的渲染图像进行多次随机扰动，再使用 HPSv2、ImageReward 等2D奖励模型评分，从而获得该视角的经验得分分布。

这种方法有明确的文献基础。Ayhan \& Berens（MIDL 2018）提出利用测试时的几何和颜色扰动估计模型输出的不确定性；ImageNet\-C 和 AugMix 则提供了系统的图像扰动类别及鲁棒性研究基础。

GitHub\+2

但需要明确：不同扰动对应不同的不确定性来源，不能简单地将所有扰动产生的得分混合，并将方差直接解释为奖励模型的置信度。

## 一、推荐的图像扰动方法

下面按照对3D资产原始内容的影响程度，将方法划分为四类。

### 光照、颜色与成像质量扰动（最推荐）

这类方法主要改变图像的像素分布，通常不改变物体的几何结构，适合分析2D奖励模型对成像条件的敏感程度。

这些方法大多可以使用 `torchvision.transforms.v2`、Albumentations 或 OpenCV 实现；ImageNet\-C 提供了噪声、模糊、数字失真等标准扰动及多级强度设置。

Zenodo\+2

### 二维几何扰动（推荐）

主要用于测试奖励模型对物体位置、尺度和画面构图的敏感性。

对于3D资产，平移和尺度扰动通常比随机裁剪、翻转更加适合作为基础分布估计手段。例如，裁剪可能遮挡缺陷，翻转可能改变物体左右结构的语义，透视变换则可能产生不符合真实3D投影的形变。

CVPR 2025 Workshop 的一项研究也观察到，不同增强操作对CLIP表征的影响存在明显差异，不能默认所有增强都会保留相同的模型响应。

CVF Open Access

### 背景与局部可见性扰动（适合辅助分析）

Random Erasing 是已有的成熟图像增强方法，但它会删除部分图像信息，因此更适合研究奖励模型对局部证据缺失的敏感程度，而不是直接估计原始视角评分的测量噪声。

AAAI

对于你的研究，前景/背景分离非常值得考虑：如果改变背景导致奖励分布大幅变化，却没有改变3D物体本身，就可能揭示奖励模型利用了与资产质量不完全相关的视觉上下文。

### 三维感知扰动（建议单独建立分布）

这一类方法需要借助3D资产重新渲染，并非严格的二维图像后处理，但与最终的3D奖励估计关系更直接。

特别注意：Camera Jitter 不应该与纯图像扰动直接混合。 相机位置变化会真正改变可见几何证据，因此对应的得分变化不仅包含评分模型的敏感性，还包含3D物体在相邻视角上的质量差异。

这一区别与 T³Bench 对多视角质量评估和视角一致性的关注是一致的。

t3bench\.com

## 二、建议的具体实现参数

下面给出一套适用于初期实验的弱扰动参数。它们是建议的起始搜索范围，并非经过验证的最优参数。

建议第一轮采用单一扰动而非直接组合多种增强，以便识别奖励变化的来源。之后再测试弱组合扰动。AugMix 可以作为组合策略的参考，但其原论文主要研究分类模型鲁棒性，并不能直接保证改善偏好奖励分布的估计。

arXiv

## 三、如何使用扰动构造每个视角的得分分布

设第 $i$ 个视角的渲染图像为 $I_i$，2D奖励模型为 $f_\phi$，文本条件为 $y$。

定义随机扰动算子 $T_\omega$，其中 $\omega\sim q(\omega)$ 是扰动参数，则：

\\\[ r\_i^\{\(m\)\}=f\_\\phi\(T\_\{\\omega\_m\}\(I\_i\),y\), \\qquad \\omega\_m\\overset\{\\mathrm\{i\.i\.d\.\}\}\{\\sim\}q\(\\omega\)\. \\\]

采样 $M$ 次后，可直接得到经验得分分布：

\\\[ \\boxed\{ \\hat P\_i\(r\)=\\frac\{1\}\{M\} \\sum\_\{m=1\}^\{M\}\\delta\(r\-r\_i^\{\(m\)\}\) \} \\\]

同时估计均值与方差：

\\\[ \\hat\\mu\_i=\\frac\{1\}\{M\}\\sum\_\{m=1\}^\{M\}r\_i^\{\(m\)\}, \\qquad \\hat\\sigma\_i^2=\\frac\{1\}\{M\-1\} \\sum\_\{m=1\}^\{M\}\(r\_i^\{\(m\)\}\-\\hat\\mu\_i\)^2\. \\\]

如果你后续使用 Wasserstein Distance 融合不同视角的分布，可以直接保留这些得分样本，不必强制假设分布服从高斯分布。

推荐第一轮实验： 对每个视角分别采样24–32个弱扰动结果，先比较颜色/光照、噪声/压缩、二维几何三组扰动的独立得分分布，然后检查采样数增加到64时，均值、方差与视角间Wasserstein距离是否稳定。

## 四、从CVPR审稿人的角度，最需要注意的问题

你需要避免一个关键的概念混淆：

\\\[ \\underbrace\{\\operatorname\{Var\}\_\{T\} \[f\_\\phi\(T\(I\_i\),y\)\]\}\_\{\\text\{扰动敏感性\}\} \\neq \\underbrace\{\\operatorname\{Var\}\(R^\\star\\mid I\_i,y\)\}\_\{\\text\{潜在真实奖励的不确定性\}\} \\\]

前者测量固定奖励模型在所选扰动分布下的响应变化；后者涉及潜在真实奖励的条件不确定性。仅使用图像增强，通常无法直接识别后者。

特别是对于HPSv2、ImageReward等偏好评分模型，模糊、遮挡、颜色改变本身就可能影响人类偏好。此时得分下降不一定意味着模型存在噪声，也可能是图像真实质量发生了改变。

因此，我更推荐你最终维护两个不同的统计对象：

- 弱扰动得分分布：反映当前视角对轻微成像与预处理变化的稳定性，用于辅助估计视角可靠性。

- 邻域视角得分分布：反映相机位置附近的3D可见证据和质量变化，用于捕捉几何缺陷、视角不一致等3D特有问题。

这两种分布恰好可以对应你之前研究中的两个核心问题：2D奖励噪声与缺失的3D几何信息。但是否能提升融合后的奖励准确性，仍需要与人工3D偏好或独立3D质量标注进行验证。

## 主要参考文献

1. [Test\-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks](https://openreview.net/references/pdf?id=rJZz-knjz&utm_source=chatgpt.com) — MIDL 2018。

2. [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261?utm_source=chatgpt.com) — ICLR 2019。

3. [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781?utm_source=chatgpt.com) — ICLR 2020。

4. [Random Erasing Data Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/7000?utm_source=chatgpt.com) — AAAI 2020。

5. [Embedding Shift Dissection on CLIP: Effects of Augmentations on VLMs](https://openaccess.thecvf.com/content/CVPR2025W/MIV/html/Dahal_Embedding_Shift_Dissection_on_CLIP_Effects_of_Augmentations_on_VLMs_CVPRW_2025_paper.html?utm_source=chatgpt.com) — CVPR Workshops 2025。

总体建议： 第一版实现优先选择轻微光度扰动、Gaussian Noise、JPEG Compression、Translation 和 Scale Jitter 五类。这些操作实现简单、计算开销可控，也比较容易建立统一的扰动强度控制。将光照重渲染和相机微扰作为独立扩展实验，会使后续奖励分布融合的数学解释更加清晰。

