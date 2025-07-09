# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:30:37
# @Function      : Inference code for human_lrm model

# import argparse
import os
# import pdb
# import time
import gc
import cv2
import numpy as np
import torch
# from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
import folder_paths
from .engine.pose_estimation.pose_estimator import PoseEstimator
from .engine.SegmentAPI.base import Bbox
from .LHM.utils.model_download_utils import AutoModelQuery

try:
    from .engine.SegmentAPI.SAM import SAM2Seg
except:
    print("\033[31mNo SAM2 found! Try using rembg to remove the background. This may slightly degrade the quality of the results!\033[0m")
    from rembg import remove

from .LHM.datasets.cam_utils import (
    build_camera_principle,
    build_camera_standard,
    create_intrinsics,
    surrounding_views_linspace,
)
from .LHM.models.modeling_human_lrm import ModelHumanLRM
from .LHM.runners import REGISTRY_RUNNERS
from . LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
)
from .LHM.utils.download_utils import download_extract_tar_from_url
from .LHM.utils.face_detector import FaceDetector

# from LHM.utils.video import images_to_video
from .LHM.utils.ffmpeg_utils import images_to_video
from .LHM.utils.hf_hub import wrap_model_hub
from .LHM.utils.logging import configure_logger
from .LHM.utils.model_card import MODEL_CARD, MODEL_CONFIG



def prior_check():
    if not os.path.exists('./pretrained_models'):
        prior_data = MODEL_CARD['prior_model']
        download_extract_tar_from_url(prior_data)


from .LHM.runners.infer.base_inferrer import Inferrer


# logger = get_logger(__name__)


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def resize_with_padding(img, target_size, padding_color=(255, 255, 255)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded


def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)
    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box


def query_model_config(model_name):
    try:
        model_params = model_name.split('-')[1]
        
        return MODEL_CONFIG[model_params] 
    except:
        return None

def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]


    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard


    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        # print(int(target_h - h))
        offset_h = max(int(target_h - h), 0)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    # if intr is not None:
    #     intr[0, 2] -= offset_x
    #     intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr


def parse_configs(base_dir):


    cfg = OmegaConf.create()
    # cli_cfg = OmegaConf.from_cli(cli_cfg)

    cfg.model_name = "LHM-1B-HF"

    # if "export_mesh" not in cli_cfg: 
    cfg.export_mesh = None
    # if "export_video" not in cli_cfg: 
    cfg.export_video= None

    query_model = AutoModelQuery()


    model_name = cfg.model_name
    model_path = query_model.query(model_name)
    #print(model_path)
    cfg.model_name = model_path 


    model_config = query_model_config(model_name)

    if model_config is not None:
        two_levels_up_dir = os.path.dirname(os.path.dirname(base_dir))
        model_config=f"{two_levels_up_dir}/custom_nodes/ComfyUI_AniCrafter/AniCrafter"+model_config
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path
        cfg.mesh_dump = os.path.join("exps", "meshs", _relative_path)  # output path

    cfg.motion_video_read_fps = 6
    # cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train



class HumanLRMInferrer(Inferrer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"
    # EXP_TYPE: str = "human_lrm_sd3"
   
    def __init__(self,base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.cfg, cfg_train = parse_configs(base_dir)
       
      
    def load_model(self):
          #print(self.cfg)
        self.facedetect = FaceDetector(
            self.base_dir + "/pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )
        self.pose_estimator = PoseEstimator(
            self.base_dir + "/pretrained_models/human_model_files/", device=avaliable_device()
        )
        
        self.parsingnet = SAM2Seg(config=os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI_AniCrafter/AniCrafter/engine/SegmentAPI/configs/sam2.1_hiera_l.yaml"))
       
        self.model: ModelHumanLRM = self._build_model(self.cfg).to(self.device)

        self.motion_dict = dict()

        

    def _build_model(self, cfg):
        from .LHM.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])

        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(
        self,
        dist_to_center: float = 2.0,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, -dist_to_center],
                    [0, 1, 0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(
            canonical_camera_extrinsics, canonical_camera_intrinsics
        )
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(
        self,
        n_views: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(
            n_views=n_views, device=device
        )
        render_camera_intrinsics = (
            create_intrinsics(
                f=0.75,
                c=0.5,
                device=device,
            )
            .unsqueeze(0)
            .repeat(render_camera_extrinsics.shape[0], 1, 1)
        )
        render_cameras = build_camera_standard(
            render_camera_extrinsics, render_camera_intrinsics
        )
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)


    def crop_face_image(self, image_path, parsing_mask=None):

        if parsing_mask is not None:
            body_rgb = np.array(Image.open(image_path))
            body_rgb = body_rgb / 255.0  # normalize to [0, 1]
            body_mask = parsing_mask / 255.0
            body_mask = (body_mask > 0.5).astype(np.float32)
            body_rgb = body_rgb[:, :, :3] * body_mask[:, :, None] + 1.0 * (1 - body_mask[:, :, None])
            rgb = np.uint8(body_rgb*255)
        else:
            rgb = np.array(Image.open(image_path))

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        bbox = self.facedetect(rgb)
        head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        head_rgb = head_rgb.permute(1, 2, 0)
        head_rgb = head_rgb.cpu().numpy()
        return head_rgb

    @torch.no_grad()
    def parsing(self, img_path):

        parsing_out = self.parsingnet._forward(np.array(Image.open(img_path)), bbox=None)

        alpha = (parsing_out.masks * 255).astype(np.uint8)

        return alpha

    def infer(self, image_path, save_gaussian_path=None,gaussian_files=None,clear_cache=True):

        gaussian_files=None if gaussian_files =="none" else gaussian_files
        shape_pose = self.pose_estimator(image_path)

        if shape_pose.ratio <= 0.1:
            print(f"body ratio is too small: {shape_pose.ratio}, continue")
            assert False

        source_size = self.cfg.source_size
        aspect_standard = 5.0 / 3

        try:
            parsing_mask = self.parsing(image_path)
        except:
            print("Error in parsing!")
            assert False
        else:
            print(f"Succeed in parsing!")

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        image = image[0]
        body_rgb = np.array(Image.open(image_path))
        body_rgb = body_rgb / 255.0  # normalize to [0, 1]
        body_mask = parsing_mask / 255.0
        body_mask = (body_mask > 0.5).astype(np.float32)
        body_rgb = body_rgb[:, :, :3] * body_mask[:, :, None] + 1.0 * (1 - body_mask[:, :, None])
        body_rgb_pil = Image.fromarray(np.uint8(body_rgb*255))
        have_head = 1

        bbox = get_bbox(parsing_mask)
        bbox_list = bbox.get_box()

        body_rgb_crop = body_rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
        body_rgb_crop_pil = Image.fromarray(np.uint8(body_rgb_crop*255))


        try:
            src_head_rgb = self.crop_face_image(image_path, parsing_mask)
            # Image.fromarray(src_head_rgb).save(save_face_path)
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            have_head = 0
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )
        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        if gaussian_files is not None and os.path.exists(gaussian_files):
            print(f"Gaussian file already exists, load it from {gaussian_files}")
            return torch.load(gaussian_files), body_rgb_pil, body_rgb_crop_pil
        else:

            device = "cuda"
            dtype = torch.float32
            shape_param = torch.tensor(shape_pose.beta, dtype=dtype).unsqueeze(0)

            smplx_params =  dict()
            # cano pose setting
            smplx_params['betas'] = shape_param.to(device)

            smplx_params['root_pose'] = torch.zeros(1,1,3).to(device)
            smplx_params['body_pose'] = torch.zeros(1,1,21, 3).to(device)
            smplx_params['jaw_pose'] = torch.zeros(1, 1, 3).to(device)
            smplx_params['leye_pose'] = torch.zeros(1, 1, 3).to(device)
            smplx_params['reye_pose'] = torch.zeros(1, 1, 3).to(device)
            smplx_params['lhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
            smplx_params['rhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
            smplx_params['expr'] = torch.zeros(1, 1, 100).to(device)
            smplx_params['trans'] = torch.zeros(1, 1, 3).to(device)

            self.model.to(dtype)

            gs_app_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(
                image.unsqueeze(0).unsqueeze(0).to(device, dtype),
                src_head_rgb.unsqueeze(0).to(device, dtype),
                None,
                None,
                None,
                None,
                None,
                smplx_params={
                    k: v.to(device) for k, v in smplx_params.items()
                },
            )
            smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose

            result = [
                gs_app_model_list[0].offset_xyz.squeeze().detach().cpu(), 
                query_points.squeeze().detach().cpu(), 
                gs_app_model_list[0].shs.squeeze().detach().cpu(), 
                gs_app_model_list[0].opacity.detach().cpu(), 
                gs_app_model_list[0].scaling.detach().cpu(), 
                gs_app_model_list[0].rotation.detach().cpu(), 
                transform_mat_neutral_pose.squeeze().detach().cpu(), 
                shape_param.squeeze().detach().cpu(), 
                shape_pose.ratio, 
                have_head
                ]
            if save_gaussian_path is not None:
                torch.save(result, save_gaussian_path)
                print(f"Gaussian file saved to {save_gaussian_path}")
        if clear_cache:
            self.clear_model_resources()  # 推理完成后调用清理方法
        return result, body_rgb_pil, body_rgb_crop_pil
    
    def clear_model_resources(self):
        # 先卸载主模型（最复杂）
        if hasattr(self, 'model'):
            # 递归卸载模型及其子模块
            def recursive_unload(module):
                # 递归处理所有子模块
                for name, child in list(module.named_children()):
                    recursive_unload(child)
                    # 解除子模块引用
                    delattr(module, name)
                
                # 清空参数和缓冲区
                if hasattr(module, '_parameters'):
                    module._parameters.clear()
                if hasattr(module, '_buffers'):
                    module._buffers.clear()
                if hasattr(module, '_modules'):
                    module._modules.clear()
                
                # 确保模块被垃圾回收
                del module
            
            try:
                recursive_unload(self.model)
            except Exception as e:
                print(f"主模型卸载错误: {e}")
            finally:
                del self.model
        
        # 再卸载其他组件
        components = [
            'parsingnet',     # SAM解析模型
            'pose_estimator', # 姿态估计模型
            'facedetect'      # 人脸检测模型
        ]
        
        for comp in components:
            if not hasattr(self, comp):
                continue
                
            obj = getattr(self, comp)
            
            # 特殊处理parsingnet
            if comp == 'parsingnet':
                if hasattr(obj, 'box_prior') and obj.box_prior is not None:
                    obj.box_prior.to('cpu')
                    del obj.box_prior
                if hasattr(obj, 'image_predictor'):
                    if hasattr(obj.image_predictor, 'model'):
                        obj.image_predictor.model.to('cpu')
                        del obj.image_predictor.model
                    del obj.image_predictor
            
            # 特殊处理pose_estimator
            elif comp == 'pose_estimator':
                if hasattr(obj, 'mhmr_model'):
                    obj.mhmr_model.to('cpu')
                    del obj.mhmr_model
            
            # 特殊处理facedetect
            elif comp == 'facedetect':
                if hasattr(obj, 'model'):
                    vgg_detector = obj.model
                    if hasattr(vgg_detector, 'model'):
                        vgg_detector.model.to('cpu')
                        del vgg_detector.model
                    del obj.model
            
            # 删除组件引用
            delattr(self, comp)
        
        # 显存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("所有模型资源已强制释放")