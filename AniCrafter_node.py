# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from torchvision.transforms import v2
from .AniCrafter.diffsynth import ModelManager_
from .node_utils import gc_cleanup,tensor2pil_upscale,tensor2pil_list,find_gaussian_files,load_images,find_directories,tensor_to_pil
#from .AniCrafter.run_pipeline_with_preprocess import prepare_models,predata_for_anicrafter,infer_anicrafter
from .AniCrafter.run_pipeline import predata_for_anicrafter_dispre,prepare_models,infer_anicrafter
import folder_paths
from .AniCrafter.diffsynth.prompters import WanPrompter_ 

########
MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
AniCrafter_weigths_path = os.path.join(folder_paths.models_dir, "AniCrafter")
if not os.path.exists(AniCrafter_weigths_path):
    os.makedirs(AniCrafter_weigths_path)
folder_paths.add_model_folder_path("AniCrafter", AniCrafter_weigths_path)

######


class AniCrafterPreImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               
                "role_image": ("IMAGE",),
                "clip_vision": (["none"] + folder_paths.get_filename_list("clip_vision"),),
                "clean_up": ("BOOLEAN", {"default": True},),
                
            }}

    RETURN_TYPES = ("AniCrafter_DATA",)
    RETURN_NAMES = ( "image_data",)
    FUNCTION = "sampler_main"
    CATEGORY = "AniCrafter"

    def sampler_main(self, role_image,clip_vision,clean_up):
        if clip_vision == "none":
           raise ValueError("Please select a CLIP model")
        else:
            clip_vision_path=folder_paths.get_full_path("clip_vision", clip_vision)

        model_manager = ModelManager_(device="cpu")
        model_manager.load_models([clip_vision_path],torch_dtype=torch.bfloat16,) # Image Encoder is loaded with float32)  #
        image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        character_image=tensor_to_pil(role_image)
        image = torch.Tensor(np.array(character_image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0).to(device)
        image_encoder.to(device)
        clip_context = image_encoder.encode_image([image])
        image_encoder.to("cpu")
        if clean_up:
            model_manager.clear_model_memory("wan_video_image_encoder")
        gc_cleanup()
        print(clip_context.shape)#torch.Size([1, 257, 1280]) is_cuda True
        #print(clip_context.dtype)#torch.bfloat16
        #clip_context=clip_context.to(device)
        
        return ({"clip_context":clip_context,"character_image":character_image,},)
    

class AniCrafterPreText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "T5": (["none"] + folder_paths.get_filename_list("clip"),),
                "cpu_infer": ("BOOLEAN", {"default": False},),
                "clean_up": ("BOOLEAN", {"default": True},),
                "prompt":("STRING", {"multiline": True,"default":"human in a scene"}),
                "negative_prompt":("STRING", {"multiline": True,"default":"细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"} ),
            }}

    RETURN_TYPES = ("AniCrafter_TEXT",)
    RETURN_NAMES = ("text_emb",)
    FUNCTION = "sampler_main"
    CATEGORY = "AniCrafter"

    def sampler_main(self, T5,cpu_infer,clean_up,prompt,negative_prompt,):
        if T5 == "none":
            raise ValueError("Please select a T5 model")
        else:
            T5_path=folder_paths.get_full_path("clip", T5)

        model_manager = ModelManager_(device="cpu")
        tokenizer_path_=os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI_AniCrafter/configs/Wan2.1-I2V-14B-720P/google/umt5-xxl")
        prompter = WanPrompter_(tokenizer_path=tokenizer_path_)
        model_manager.load_models([T5_path,],torch_dtype=torch.bfloat16,) # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.)
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        #print(text_encoder_model_and_path)
        if text_encoder_model_and_path is not None:
            text_encoder, tokenizer_path = text_encoder_model_and_path

            prompter.fetch_models(text_encoder)
            prompter.fetch_tokenizer(tokenizer_path_)
        if not cpu_infer:
            prompter.to_cuda()
            prompt_emb_posi = prompter.encode_prompt(prompt, positive=True,)
            prompt_emb_nega = prompter.encode_prompt(negative_prompt, positive=False,)
            prompter.offload()
        else:
            prompt_emb_posi = prompter.encode_prompt(prompt, positive=True,device="cpu")
            prompt_emb_nega = prompter.encode_prompt(negative_prompt, positive=False,device="cpu")
            prompt_emb_posi = prompt_emb_posi.to(device)
            prompt_emb_nega = prompt_emb_nega.to(device)
        if clean_up:
            model_manager.clear_model_memory("wan_video_text_encoder")
        print(prompt_emb_posi.shape,prompt_emb_nega.shape,prompt_emb_posi.is_cuda,prompt_emb_posi.dtype) #torch.Size([1, 512, 4096]) torch.Size([1, 512, 4096]) # True torch.float32

       
        
        return ({"prompt_emb_posi": {"context": prompt_emb_posi}, "prompt_emb_nega": {"context": prompt_emb_nega}, },)


class AniCrafterPreVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        gaussian_files_list = ["none"] + find_gaussian_files(folder_paths.get_input_directory()) if find_gaussian_files(folder_paths.get_input_directory()) else ["none"]
        pre_video_dir_list = ["none"]+find_directories(folder_paths.get_output_directory()) if find_directories(folder_paths.get_output_directory()) else ["none"]
        return {
            "required": {
                "image_data":("AniCrafter_DATA",),
                "video_image": ("IMAGE",),
                "gaussian_files": (gaussian_files_list,),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "display": "number"}),
                "max_frames": ("INT", {"default": 80, "min": 8, "max": 2048, "step": 4, "display": "number"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 5.0, "max": 120.0, "step": 1.0}),
                "camera_fov":("INT", {"default": 60, "min": 10, "max": 100, "step": 5, "display": "number"}),
                "clean_up": ("BOOLEAN", {"default": True},),
                "preprocess_input": ("BOOLEAN", {"default": True},),
                "pre_video_dir": (pre_video_dir_list,),
            },
             "optional": {
                "video_mask": ("IMAGE",),  # [B,H,W,C], C=3,B>1
                "video_bkgd": ("IMAGE",),   
            }
            }

    RETURN_TYPES = ("AniCrafter_PREDATA",)
    RETURN_NAMES = ("data_dict", )
    FUNCTION = "sampler_main"
    CATEGORY = "AniCrafter"

    def sampler_main(self, image_data,video_image,gaussian_files,width,height ,max_frames,fps,camera_fov,clean_up,preprocess_input,pre_video_dir,**kwargs ):
        
        input_mask=kwargs.get("video_mask")
        input_bkgd=kwargs.get("video_bkgd")

        character_image=image_data.get("character_image")
        max_frames=max_frames + 1 #  must be 1 (mod 4)

        
        use_input_mask=tensor2pil_list(input_mask,width,height)[:max_frames] if isinstance(input_mask, torch.Tensor) else None
        use_bkgd_video=tensor2pil_list(video_image,width,height)[:max_frames] if isinstance(input_bkgd, torch.Tensor) else None

       
        image_list=tensor2pil_list(video_image,width,height)[:max_frames]
        #print("image_list",len(image_list))

        if use_input_mask is not None:
            if len(image_list) != len(use_input_mask): #防止mask和视频长度不一致
                min_frames=min(len(image_list),len(use_input_mask))
                image_list=image_list[:min_frames]
                use_input_mask=use_input_mask[:min_frames]
                max_frames=min_frames

        if use_bkgd_video is not None:
            if len(image_list) != len(use_bkgd_video): #防止mask，内绘图和视频长度不一致
                min_frames=min(len(image_list),len(use_bkgd_video),len(use_input_mask)) if  use_input_mask is not None else min(len(image_list),len(use_bkgd_video))
                use_bkgd_video=use_bkgd_video[:min_frames]
                use_input_mask=use_input_mask[:min_frames] if use_input_mask is not None else None
                image_list=image_list[:min_frames]
                max_frames=min_frames


        frame_process_norm = v2.Compose([
        v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
         ])
        if pre_video_dir!="none":
            pre_video_dir_=os.path.join(folder_paths.get_output_directory(),pre_video_dir)
        else:
            pre_video_dir_=pre_video_dir
        ref_combine_blend_tensor,ref_combine_smplx_tensor,height_, width_=predata_for_anicrafter_dispre(frame_process_norm,
                                image_list,character_image,AniCrafter_weigths_path,gaussian_files,clean_up,preprocess_input,pre_video_dir_,use_input_mask,use_bkgd_video,fps,camera_fov,max_frames)
        image_data["ref_combine_blend_tensor"]=ref_combine_blend_tensor
        image_data["ref_combine_smplx_tensor"]=ref_combine_smplx_tensor
        image_data["width"]=width_
        image_data["height"]=height_
        image_data["max_frames"]=max_frames
        image_data["fps"]=fps
        return (image_data,)

class AniCrafterLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit": (["none"] + folder_paths.get_filename_list("diffusion_models"),),
                "vae": (["none"] + folder_paths.get_filename_list("vae"),),
                "lora_alpha":("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "use_mmgp": ([ "LowRAM_LowVRAM","VerylowRAM_LowVRAM","LowRAM_HighVRAM","HighRAM_LowVRAM","HighRAM_HighVRAM","none",],),

            },
        }

    RETURN_TYPES = ("MODEL_AniCrafter",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "AniCrafter"

    def loader_main(self,dit,vae,lora_alpha,use_mmgp):
        wan_repo="Wan2.1-I2V-14B-720P"
        if dit == "none":
            raise ValueError("Please select a DIT model")
        else:
            dit_path=folder_paths.get_full_path("diffusion_models", dit)
            if "480P" in dit_path:
                wan_repo="Wan2.1-I2V-14B-480P"
        
        if vae == "none":
            raise ValueError("Please select a VAE model")
        else:
            vae_path=folder_paths.get_full_path("vae", vae)
       

        # load model
        print("***********Load model ***********")

        pipe = prepare_models(dit_path,vae_path, os.path.join(AniCrafter_weigths_path, "pretrained_models/anicrafter"),lora_alpha)
        if use_mmgp!="none":
            from mmgp import offload, profile_type
            if use_mmgp=="VerylowRAM_LowVRAM":
                offload.profile({"transformer": pipe.dit, "vae": pipe.vae}, profile_type.VerylowRAM_LowVRAM)
            elif use_mmgp=="LowRAM_LowVRAM":  
                offload.profile({"transformer": pipe.dit, "vae": pipe.vae}, profile_type.LowRAM_LowVRAM)
            elif use_mmgp=="LowRAM_HighVRAM":
                offload.profile({"transformer": pipe.dit, "vae": pipe.vae}, profile_type.LowRAM_HighVRAM)
            elif use_mmgp=="HighRAM_LowVRAM":
                offload.profile({"transformer": pipe.dit, "vae": pipe.vae}, profile_type.HighRAM_LowVRAM)
            elif use_mmgp=="HighRAM_HighVRAM":
                offload.profile({"transformer": pipe.dit, "vae": pipe.vae}, profile_type.HighRAM_HighVRAM)
            
        print("***********Load model done ***********")

        gc_cleanup()

        return ({"pipe":pipe ,"wan_repo":wan_repo},)



class AniCrafterSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_emb": ("AniCrafter_TEXT",),
                "data_dict": ("AniCrafter_PREDATA",),  # {}
                "model": ("MODEL_AniCrafter",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 2048, "step": 1, "display": "number"}),
                "cfg_value": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "use_teacache": ("BOOLEAN", {"default": True},),
                "use_tiled": ("BOOLEAN", {"default": True},),
            }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "sampler_main"
    CATEGORY = "AniCrafter"

    def sampler_main(self,text_emb, data_dict, model, seed, num_inference_steps,cfg_value, use_teacache,use_tiled,):

        wan_repo=model.get("wan_repo")
        model=model.get("pipe")
        print("***********Start infer  ***********")

        iamges = infer_anicrafter(model, data_dict.get("ref_combine_blend_tensor"),data_dict.get("ref_combine_smplx_tensor"),
                                 data_dict.get("height"),data_dict.get("width"),
                                 num_inference_steps,seed ,use_teacache,cfg_value,use_tiled,text_emb,data_dict,wan_repo)
        gc.collect()
        torch.cuda.empty_cache()
        return (load_images(iamges), data_dict.get("fps"))


NODE_CLASS_MAPPINGS = {
    "AniCrafterPreImage": AniCrafterPreImage,
    "AniCrafterPreText": AniCrafterPreText,
    "AniCrafterPreVideo": AniCrafterPreVideo,
    "AniCrafterLoader": AniCrafterLoader,
    "AniCrafterSampler": AniCrafterSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AniCrafterPreImage": "AniCrafterPreImage",
    "AniCrafterPreText": "AniCrafterPreText",
    "AniCrafterPreVideo": "AniCrafterPreVideo",
    "AniCrafterLoader": "AniCrafterLoader",
    "AniCrafterSampler": "AniCrafterSampler",
}
