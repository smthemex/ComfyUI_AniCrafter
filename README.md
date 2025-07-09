# ComfyUI_AniCrafter
[AniCrafter](https://github.com/MyNiuuu/AniCrafter): Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models, you can try this methods  when use ComfyUI.

# Tips 
* 当前代码仅能跑通demo测试的输入视频，mask和splm的预处理还没整好。pre_video的高斯pth 第一次使用先选择none，测试显存12G。
* only demo input now

# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_AniCrafter.git
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```
and 
```
pip install ninja
pip install flash-attn --no-build-isolation
pip install tb-nightly
pip install git+https://github.com/XPixelGroup/BasicSR
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/hitsz-zuoqi/sam2/
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
pip install git+https://github.com/camenduru/simple-knn/

```

# If python version >3.10,need modify 'chumpy' packeage  :
* '...site-packages\chumpy\ch.py' ,line 1203 ,change ' inspect.getargspec' to 'inspect.getfullargspec'
* '...site-packages\chumpy\__init__.py' ,line 11,change  'from numpy import bool, int, float, complex, object, unicode, str, nan, inf 'to
```
from numpy import complex_, object_, nan, inf
import builtins
bool_ = builtins.bool
int_ = builtins.int
float_ = builtins.float
str_ = builtins.str
```


# Example
![](https://github.com/smthemex/ComfyUI_AniCrafter/blob/main/example_workflows/example.gif)


#Citation
```
@article{niu2025anicrafter,
  title={AniCrafter: Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models},
  author={Niu, Muyao and Cao, Mingdeng and Zhan, Yifan and Zhu, Qingtian and Ma, Mingze and Zhao, Jiancheng and Zeng, Yanhong and Zhong, Zhihang and Sun, Xiao and Zheng, Yinqiang},
  journal={arXiv preprint arXiv:2505.20255},
  year={2025}
}
```
