# ComfyUI_AniCrafter
[AniCrafter](https://github.com/MyNiuuu/AniCrafter): Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models, you can try this methods  when use ComfyUI.

# Update 
* 0715 修复无法重复推理的bug，修复高斯np文件加载device错误，修复输入图片自动裁切错误,example 新增测试用的华仔及其np文件和美少女跳舞内容；
* 支持自定义视频的推理，支持预处理视频（mask，背景内绘及smplx剪辑）和json文件的 以及gaussian.pth的复用（首次生成需要选择none）；为避免人脸失真，推理尺寸越大越好。等下个周末再修复一些bug吧。
* need another weekend to fix bugs

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
pip install mmcv_full-1.7.2
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
* 3.1.1 [MyNiuuu/Anicrafter_release](https://huggingface.co/MyNiuuu/Anicrafter_release/tree/main) all fiels/下载pretrained_models所有文件，保存文件夹结构
* 3.1.2 [propainter ](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/) download/下载地址；
```
├── your comfyUI/models/AniCrafter/
|   ├──pretrained_models
|       ├── all fiels  #pretrained_models目录下所有文件及文件结构
|       ├── propainter
|           ├── ProPainter.pth #propainter models 下次改成可预加载
|           ├── raft-things.pth
|           ├── recurrent_flow_completion.pth
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
![](https://github.com/smthemex/ComfyUI_AniCrafter/blob/main/example_workflows/example0715.gif)


#Citation
```
@article{niu2025anicrafter,
  title={AniCrafter: Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models},
  author={Niu, Muyao and Cao, Mingdeng and Zhan, Yifan and Zhu, Qingtian and Ma, Mingze and Zhao, Jiancheng and Zeng, Yanhong and Zhong, Zhihang and Sun, Xiao and Zheng, Yinqiang},
  journal={arXiv preprint arXiv:2505.20255},
  year={2025}
}
```
