# ComfyUI_AniCrafter
[AniCrafter](https://github.com/MyNiuuu/AniCrafter): Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models, you can try this methods  when use ComfyUI.

# Tips 
* 当前代码仅能跑通demo测试的输入视频，mask和splm的预处理还没整好。pre_video的高斯pth 第一次使用先选择none，测试显存12G，基于mmgp库的优秀调度； 
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

# 3  Models
* 3.1 [MyNiuuu/Anicrafter_release](https://huggingface.co/MyNiuuu/Anicrafter_release/tree/main) all fiels/下载pretrained_models所有文件，保存文件夹结构
```
├── your comfyUI/models/AniCrafter/
|   ├──pretrained_models
|       ├── all fiels  #pretrained_models目录下所有文件及文件结构
```
* 3.2 [Wan-AI/Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main) download clip,clipvison and vae /下载clip，clipvison 和vae  
```
├── your comfyUI/models/vae/
|   ├──Wan2.1_VAE.pth
├── models/clip_vision/
|   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
├── models/clip/
|   ├── models_t5_umt5-xxl-enc-bf16.pth
```
* 3.3 gfpgan auto download/gfpgan 自动下载
* 3.4 Wan2.1-I2V-14B-720P single model  from here [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main) 下载KJ的单体wan模型，全量跑不动
```
├── your comfyUI/models/diffusion_models/
|   ├──Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors  #kj 16G
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
