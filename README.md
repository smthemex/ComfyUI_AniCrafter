# ComfyUI_AniCrafter
[AniCrafter](https://github.com/MyNiuuu/AniCrafter): Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models, you can try this methods  when use ComfyUI.

# Update 
* 0721 适配lightX2V的 LCM调度器（4步，代码直接从lightX2V获取） UniPC调度器(10步，代码直接从KJ那拿的)，修改预处理视频文件列表加载逻辑，去掉bgkb视频的必要条件。
* 0719 pm 新增lightX2V lora加载和常规style lora加载菜单，如果使用lightX2V lora，cfg请调整为1，推荐步数6步，风格lora需要填写prompt；
* 0719 新增镜头fov参数，全身镜头默认60，对于广角镜头需要调节至小于45，否则smplx的人物占比会变小，模型支持480P；
* 0718 新增mmgp模式可选，高GPU和VRAM 可以选none或high模式，新增bgkb视频可选输入（背景去掉人物内绘，推荐用插件[ComfyUI_DiffuEraser](https://github.com/smthemex/ComfyUI_DiffuEraser)），请使用最新的workflow.
* 支持自定义视频的推理，支持预处理视频（mask，背景内绘及smplx剪辑）和json文件的 以及gaussian.pth的复用（首次生成需要选择none）；为避免人脸失真，推理尺寸越大越好（使用720P模型时）。
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
```
[mmcv install error look this ](https://gitee.com/Wilson_Lws/MuseTalk-50Series-Adaptation/blob/master/README.md)
```
pip install flash-attn --no-build-isolation
pip install tb-nightly
pip install git+https://github.com/XPixelGroup/BasicSR
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/hitsz-zuoqi/sam2/
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/
pip install git+https://github.com/camenduru/simple-knn/

```

# Install TIPS
* 2.1 如果python版本大于3.9 因为numpy的原因，需要手动修改chumpy库如下/If python version >3.9,need modify 'chumpy' packeage  ：
Path '...site-packages/chumpy/ch.py, line 1203 ,change 修改参数如下
```
inspect.getargspec
to
inspect.getfullargspec
```
* 2.2  init.py 修改参数如下 :
```
.../site-packages/chumpy/__init__.py  #line 11,change
#from numpy import bool, int, float, complex, object, unicode, str, nan, inf

from numpy import complex_, object_, nan, inf
import builtins
bool_ = builtins.bool
int_ = builtins.int
float_ = builtins.float
str_ = builtins.str
```
* 2.3  ...Lib\site-packages\mmcv\device\npu\data_parallel.py line 20  data_parallel报错时，打开文件，修改如下（就是加个list）
```
#for m in sys.modules:
for m in list(sys.modules):
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
* 3.4 Wan2.1-I2V-14B-720P  or Wan2.1-I2V-14B-480P  single model  from here [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main) 下载KJ的单体wan模型  
```
├── your comfyUI/models/diffusion_models/
|   ├──Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors  # or  Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors  #16G
```
* 3.4 wan 2.1 lora lightX2V or other from here [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main)  支持加速lora
```
  ├── your comfyUI/models/loras/
     ├──lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors # or other 
```

# Example
* new use lightx2v
![](https://github.com/smthemex/ComfyUI_AniCrafter/blob/main/example_workflows/example0721.png)
*   version
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
