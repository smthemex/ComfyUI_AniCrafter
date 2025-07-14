# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-19 12:47:58
# @Function      : video motion process pipeline
import copy
import json
import os
import sys
from omegaconf import OmegaConf

# current_dir_path = os.path.dirname(__file__)
# sys.path.append(current_dir_path + "/../pose_estimation")
import argparse
import copy
import gc
import json
import os
import random
import sys
import time

from PIL import Image

import cv2
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from .blocks import SMPL_Layer
from .blocks.detector import DetectionModel
from .model import forward_model, load_model
from .pose_utils.constants import KEYPOINT_THR
from .pose_utils.image import img_center_padding, normalize_rgb_tensor
from .pose_utils.inference_utils import get_camera_parameters
from .pose_utils.postprocess import OneEuroFilter, smplx_gs_smooth
from .pose_utils.render import render_video, get_render_SMPLX_frames
from .pose_utils.tracker import bbox_xyxy_to_cxcywh, track_by_area
from .smplify import TemporalSMPLify

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)




class Bbox:
    def __init__(self, box, mode="whwh"):

        assert len(box) == 4
        assert mode in ["whwh", "xywh"]
        self.box = box
        self.mode = mode

    def to_xywh(self):

        if self.mode == "whwh":

            l, t, r, b = self.box

            center_x = (l + r) / 2
            center_y = (t + b) / 2
            width = r - l
            height = b - t
            return Bbox([center_x, center_y, width, height], mode="xywh")
        else:
            return self

    def to_whwh(self):

        if self.mode == "whwh":
            return self
        else:

            cx, cy, w, h = self.box
            l = cx - w // 2
            t = cy - h // 2
            r = cx + w - (w // 2)
            b = cy + h - (h // 2)

            return Bbox([l, t, r, b], mode="whwh")

    def area(self):

        box = self.to_xywh()
        _, __, w, h = box.box

        return w * h

    def get_box(self):
        return list(map(int, self.box))

    def scale(self, scale, width, height):
        new_box = self.to_xywh()
        cx, cy, w, h = new_box.get_box()
        w = w * scale
        h = h * scale

        l = cx - w // 2
        t = cy - h // 2
        r = cx + w - (w // 2)
        b = cy + h - (h // 2)

        l = int(max(l, 0))
        t = int(max(t, 0))
        r = int(min(r, width))
        b = int(min(b, height))

        return Bbox([l, t, r, b], mode="whwh")

    def __repr__(self):
        box = self.to_whwh()
        l, t, r, b = box.box

        return f"BBox(left={l}, top={t}, right={r}, bottom={b})"



def get_bbox(mask):
    height, width = mask.shape

    # obtain bbox
    _h, _w = np.where(mask == 1)

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




def video_to_pil_images(video_path, height=None, width=None):
    if video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        pil_images = []
        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break  # 视频结束或读取失败
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if height is not None and width is not None:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
        cap.release()
    elif os.path.isdir(video_path):
        frame_files = sorted([os.path.join(video_path, x) for x in os.listdir(video_path) if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')])
        pil_images = []
        for frame in frame_files:
            frame = cv2.imread(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if height is not None and width is not None:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
    else:
        raise ValueError("Unsupported video format. Please provide a .mp4 file or a directory of images.")
    return pil_images




def load_video(video_path, pad_ratio):
    fps = 30
    all_images = video_to_pil_images(video_path)

    try:
        all_masks = video_to_pil_images(video_path.replace('raw_video', 'mask_video'))
    except:
        print('error in reading masks, using frames only')
        use_mask = False
    else:
        use_mask = True
        
    frames = []
    for idx, img in enumerate(all_images):
        frame = cv2.cvtColor(np.array(all_images[idx]), cv2.COLOR_BGR2RGB)
        if use_mask:
            mask = np.array(all_masks[idx]) / 255.
            bbox = get_bbox(mask[:, :, 0])
            bbox_list = bbox.get_box()
            mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]] = 1
            frame = np.uint8(frame * mask)
        # since the tracker and detector receive BGR images as inputs
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if pad_ratio > 0:
            frame, offset_w, offset_h = img_center_padding(frame, pad_ratio)
        else:
            offset_w, offset_h = 0, 0
        frames.append(frame)
    height, weight, _ = frames[0].shape
    return frames, height, weight, fps, offset_w, offset_h, all_images


def pre_pil_mask(all_images, pad_ratio,all_masks=None):
    #fps = 30
    #all_images = video_to_pil_images(video_path)
    # for i, image in enumerate(all_images):
    #     image.save(f'{i}_.png')
    if all_masks is None:
        print('error in reading masks, using frames only')
        use_mask = False
        
    else:
        use_mask = True
        processed_masks = []
        for i, mask in enumerate(all_masks):
            if isinstance(mask, np.ndarray):
                # 将numpy数组转换为灰度图
                if mask.ndim == 2:  # [H, W]
                    pass  # 已经是灰度图，无需转换
                elif mask.ndim == 3 and mask.shape[2] == 1:  # [H, W, 1]
                    mask = mask.squeeze(-1)  # 去除多余维度
                elif mask.ndim == 3 and mask.shape[2] in [3, 4]:  # [H, W, 3] 或 [H, W, 4]
                    if mask.dtype == np.uint8:
                        # 使用OpenCV的颜色空间转换函数将图像转换为灰度图
                        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY if mask.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
                    else:
                        # 非uint8类型，先转换为uint8再处理
                        mask = (mask * 255).astype(np.uint8)
                        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY if mask.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
                else:
                    raise ValueError(f"Unsupported mask shape: {mask.shape}")
            elif isinstance(mask, Image.Image):
                # 确保PIL图像是灰度图
                if mask.mode != 'L':
                    mask = mask.convert('L')
                mask = np.array(mask)  # 转换为numpy数组
            else:
                raise TypeError(f"Unsupported mask type: {type(mask)}")
            processed_masks.append(mask)

    frames = []
    for idx, img in enumerate(all_images):
        frame = cv2.cvtColor(np.array(all_images[idx]), cv2.COLOR_BGR2RGB)
        if use_mask:
            mask = np.array(processed_masks[idx], dtype=np.float32) / 255.
            bbox = get_bbox(mask)
            bbox_list = bbox.get_box()
            mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]] = 1
            mask = np.expand_dims(mask, axis=-1)
            frame = np.uint8(frame * mask)
            # mask = np.array(processed_masks[idx]) / 255.
            # bbox = get_bbox(mask[:, :, 0])
            # bbox_list = bbox.get_box()
            # mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]] = 1
            # frame = np.uint8(frame * mask)
        # since the tracker and detector receive BGR images as inputs
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if pad_ratio > 0:
            frame, offset_w, offset_h = img_center_padding(frame, pad_ratio)
        else:
            offset_w, offset_h = 0, 0
        frames.append(frame)
    height, width, _ = frames[0].shape
    return frames, height, width,  offset_w, offset_h, all_images




def images_crop(images, bboxes, target_size, device=torch.device("cuda")):
    # bboxes: cx, cy, w, h
    crop_img_list = []
    crop_annotations = []
    i = 0
    raw_img_size = max(images[0].shape[:2])
    for img, bbox in zip(images, bboxes):

        left = max(0, int(bbox[0] - bbox[2] // 2))
        right = min(img.shape[1] - 1, int(bbox[0] + bbox[2] // 2))
        top = max(0, int(bbox[1] - bbox[3] // 2))
        bottom = min(img.shape[0] - 1, int(bbox[1] + bbox[3] // 2))
        crop_img = img[top:bottom, left:right]
        crop_img = torch.Tensor(crop_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)

        _, _, h, w = crop_img.shape
        scale_factor = min(target_size / w, target_size / h)
        crop_img = F.interpolate(crop_img, scale_factor=scale_factor, mode="bilinear")

        _, _, h, w = crop_img.shape
        pad_left = (target_size - w) // 2
        pad_top = (target_size - h) // 2
        pad_right = target_size - w - pad_left
        pad_bottom = target_size - h - pad_top
        crop_img = F.pad(
            crop_img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(crop_img)

        crop_img_list.append(resize_img)
        crop_annotations.append(
            (
                left,
                top,
                pad_left,
                pad_top,
                scale_factor,
                target_size / scale_factor,
                raw_img_size,
            )
        )

    return crop_img_list, crop_annotations


def generate_pseudo_idx(keypoints, patch_size, n_patch, crop_annotation):

    device = keypoints.device
    anchors = torch.stack([keypoints[3], keypoints[4], keypoints[5], keypoints[6]])

    mask = anchors[..., -1] >= KEYPOINT_THR
    if mask.sum() < 2:

        return None, None
    anchors = anchors[mask, :2]  # N, 2

    radius = torch.norm(anchors.max(dim=0)[0] - anchors.min(dim=0)[0]) / 2

    head_pseudo_loc = anchors.mean(0)
    if crop_annotation is not None:
        left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = (
            crop_annotation
        )
        head_pseudo_loc = (
            head_pseudo_loc - torch.tensor([left, top], device=device)
        ) * scale_factor + torch.tensor([pad_left, pad_top], device=device)
        radius = radius * scale_factor
    coarse_loc = (head_pseudo_loc // patch_size).int()  # (nhv,2)
    pseudo_idx = torch.clamp(coarse_loc, 0, n_patch - 1)  # (nhv,2)
    pseudo_idx = (
        torch.zeros((1,), dtype=torch.int32, device=device),
        pseudo_idx[1:2],
        pseudo_idx[0:1],
        torch.zeros((1,), dtype=torch.int32, device=device),
    )
    max_dist = (radius // patch_size).int()
    if max_dist < 2:
        max_dist = None
    return pseudo_idx, max_dist


def project2origin_img(target_human, crop_annotation):
    if target_human is None:
        return target_human
    left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = crop_annotation
    device = target_human["loc"].device

    target_human["loc"] = (
        target_human["loc"] - torch.tensor([pad_left, pad_top], device=device)
    ) / scale_factor + torch.tensor([left, top], device=device)

    target_human["dist"] = target_human["dist"] / (crop_size / raw_size)
    return target_human


def empty_frame_pad(pose_results):
    if len(pose_results) == 1:
        return pose_results
    all_is_None = True
    for i in range(1, len(pose_results)):
        if pose_results[i] is None and pose_results[i - 1] is not None:
            # print(i)
            pose_results[i] = copy.deepcopy(pose_results[i - 1])
        if pose_results[i] is not None:
            all_is_None = False
    if all_is_None:
        return []
    for i in range(len(pose_results) - 2, -1, -1):
        if pose_results[i] is None and pose_results[i + 1] is not None:
            pose_results[i] = copy.deepcopy(pose_results[i + 1])
    return pose_results


def parse_chunks(
    frame_ids,
    pose_results,
    k2d,
    bboxes,
    min_len=10,
):
    """If a track disappear in the middle,
    we separate it to different segments
    """
    data_chunks = []
    if isinstance(frame_ids, list):
        frame_ids = np.array(frame_ids)
    step = frame_ids[1:] - frame_ids[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]
    start = 0
    for bk in breaks[1:]:
        f_chunk = frame_ids[start:bk]

        if len(f_chunk) >= min_len:
            data_chunk = {
                "frame_id": f_chunk,
                "keypoints_2d": k2d[start:bk],
                "bbox": bboxes[start:bk],
                "rotvec": [],
                "beta": [],
                "loc": [],
                "dist": [],
            }
            padded_pose_results = empty_frame_pad(pose_results[start:bk])

            for pose_result in padded_pose_results:
                data_chunk["rotvec"].append(pose_result["rotvec"])
                data_chunk["beta"].append(pose_result["shape"])
                data_chunk["loc"].append(pose_result["loc"])
                data_chunk["dist"].append(pose_result["dist"])
            if len(padded_pose_results) > 0:
                data_chunks.append(data_chunk)
        start = bk

    start = breaks[-1]  # last chunk
    bk = len(frame_ids)
    f_chunk = frame_ids[start:bk]

    if len(f_chunk) >= min_len:
        data_chunk = {
            "frame_id": f_chunk,
            "keypoints_2d": k2d[start:bk].clone().detach(),
            "bbox": bboxes[start:bk].clone().detach(),
            "rotvec": [],
            "beta": [],
            "loc": [],
            "dist": [],
        }
        padded_pose_results = empty_frame_pad(pose_results[start:bk])
        for pose_result in padded_pose_results:
            data_chunk["rotvec"].append(pose_result["rotvec"])
            data_chunk["beta"].append(pose_result["shape"])
            data_chunk["loc"].append(pose_result["loc"])
            data_chunk["dist"].append(pose_result["dist"])

        if len(padded_pose_results) > 0:

            data_chunks.append(data_chunk)

    for data_chunk in data_chunks:
        for key in ["rotvec", "beta", "loc", "dist"]:
            try:
                data_chunk[key] = torch.stack(data_chunk[key])
            except:
                print(key)

    return data_chunks


def load_models(model_path, device):
    ckpt_path = os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt")
    pose_model = load_model(ckpt_path, model_path, device=device)
    print("load hmr")
    pose_model_ckpt = os.path.join(
        model_path, "pose_estimate", "vitpose-h-wholebody.pth"
    )
    keypoint_detector = DetectionModel(pose_model_ckpt, device)
    print("load detection")
    smplx_model = SMPL_Layer(
        model_path,
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    print("load smplx")
    return pose_model, keypoint_detector, smplx_model


class Video2MotionPipeline:
    def __init__(
        self,
        model_path,
        device,
        kp_mode="vitpose",
        visualize=True,
        pad_ratio=0.,
        fov=60,
        fps=30,
    ):
        self.device = device
        self.visualize = visualize
        self.kp_mode = kp_mode
        self.pad_ratio = pad_ratio
        self.fov = fov
        self.fps = fps
        self.pose_model, self.keypoint_detector, self.smplx_model = load_models(
            model_path, self.device
        )
        self.smplx_model.to(self.device)
        self.smplify = TemporalSMPLify(
            smpl=self.smplx_model, device=self.device, num_steps=50
        )

    def track(self, all_frames):
        self.keypoint_detector.initialize_tracking()
        for frame in all_frames:
            self.keypoint_detector.track(frame, self.fps, len(all_frames))
        tracking_results = self.keypoint_detector.process(self.fps)
        # note: only surpport pose estimation for one character
        main_character = None
        max_frame_length = -1
        for _id in tracking_results.keys():
            if len(tracking_results[_id]["frame_id"]) > max_frame_length:
                main_character = _id

        bboxes = tracking_results[main_character]["bbox"]
        frame_ids = tracking_results[main_character]["frame_id"]
        frames = [all_frames[i] for i in frame_ids]
        assert not (bboxes[0][0] == 0 and bboxes[0][2] == 0)

        return bboxes, frame_ids, frames

    def detect_keypoint2d(self, bboxes, frames):
        if self.kp_mode == "vitpose":
            keypoints, bboxes = self.keypoint_detector.batch_detection(bboxes, frames)
        else:
            raise NotImplementedError
        return bboxes, keypoints

    def estimate_pose(self, frame_ids, frames, keypoints, bboxes, raw_K, video_length):
        target_img_size = self.pose_model.img_size
        patch_size = self.pose_model.patch_size

        K = get_camera_parameters(
            target_img_size, fov=self.fov, p_x=None, p_y=None, device=self.device
        )

        keypoints = torch.tensor(keypoints, device=self.device)
        bboxes = torch.tensor(bboxes, device=self.device)
        bboxes = bbox_xyxy_to_cxcywh(bboxes, scale=1.5)

        crop_images, crop_annotations = images_crop(
            frames, bboxes, target_size=target_img_size, device=self.device
        )

        all_frame_results = []
        # model inference
        for i, image in enumerate(crop_images):

            # Calculate the possible search area for the primary joint (head) based on 2D keypoints
            # pseudo_idx: The index of the search area center after patching
            # max_dist: The maximum radius of the search area
            pseudo_idx, max_dist = generate_pseudo_idx(
                keypoints[i],
                patch_size,
                int(target_img_size / patch_size),
                crop_annotations[i],
            )
            humans = forward_model(
                self.pose_model, image, K, pseudo_idx=pseudo_idx, max_dist=max_dist
            )
            target_human = track_by_area(humans, target_img_size)
            target_human = project2origin_img(target_human, crop_annotations[i])

            all_frame_results.append(target_human)

        # parse chunk & missed frame padding
        data_chunks = parse_chunks(
            frame_ids,
            all_frame_results,
            keypoints,
            bboxes,
            min_len=int(self.fps / 10),
        )

        trans_cam_fill = np.zeros((video_length, 3))
        smpl_poses_cam_fill = np.zeros((video_length, 55, 3))
        smpl_shapes_fill = np.zeros((video_length, 10))
        all_verts = [None] * video_length
        for data_chunk in data_chunks:
            # one_euro filter on 2d keypoints

            one_euro = OneEuroFilter(
                min_cutoff=1.2, beta=0.3, sampling_rate=self.fps, device=self.device
            )
            for i in range(len(data_chunk["keypoints_2d"])):
                data_chunk["keypoints_2d"][i, :2] = one_euro.filter(
                    data_chunk["keypoints_2d"][i, :2]
                )
    
            poses, betas, transl = self.smplify.fit(
                data_chunk["rotvec"],
                data_chunk["beta"],
                data_chunk["dist"],
                data_chunk["loc"],
                raw_K,
                data_chunk["keypoints_2d"],
                data_chunk["bbox"],
            )

            # gaussian filter
            with torch.no_grad():

                poses, betas, transl = smplx_gs_smooth(
                    poses, betas, transl, fps=self.fps
                )

                out = self.smplx_model(
                    poses,
                    betas,
                    None,
                    None,
                    transl=transl,
                    K=raw_K,
                    expression=None,
                    rot6d=False,
                )

                transl = out["transl_pelvis"].squeeze(1)
                poses_ = self.smplx_model.convert_standard_pose(poses)
                smpl_poses_cam_fill[data_chunk["frame_id"]] = poses_.cpu().numpy()
                smpl_shapes_fill[data_chunk["frame_id"]] = betas.cpu().numpy()
                trans_cam_fill[data_chunk["frame_id"]] = transl.cpu().numpy()

            for i, frame_id in enumerate(data_chunk["frame_id"]):
                try:
                    if all_verts[frame_id] is None:
                        all_verts[frame_id] = []
                    all_verts[frame_id].append(out["v3d"][i])
                except:
                    break

        return smpl_poses_cam_fill, smpl_shapes_fill, trans_cam_fill, all_verts

    def save_video(
        self, all_frames, frame_ids, bboxes, keypoints, verts, K, out_folder
    ):
        all_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in all_frames]
        save_name = os.path.join(out_folder, "pose_visualized.mp4")

        # 2d keypoints visualization
        for i, frame_id in enumerate(frame_ids):
            keypoint_results = [{"bbox": bboxes[i], "keypoints": keypoints[i]}]

            all_frames[frame_id] = self.keypoint_detector.visualize(
                all_frames[frame_id], keypoint_results
            )

        render_video(
            verts,
            self.pose_model.smpl_layer["neutral_10"].bm_x.faces,
            K,
            all_frames,
            self.fps,
            save_name,
            self.device,
            True,
        )

    def render_SMPLX_frame(
        self, all_frames, verts, K
    ):
        all_SMPLX_frames = get_render_SMPLX_frames(
            verts,
            self.pose_model.smpl_layer["neutral_10"].bm_x.faces,
            K,
            all_frames,
            self.device
        )

        return all_SMPLX_frames

    def save_results(self, out_path, frame_ids, poses, betas, transl, K, img_wh):
        K = K[0].cpu().numpy()
        for i in frame_ids:

            smplx_param = {}
            smplx_param["betas"] = betas[i].tolist()
            smplx_param["root_pose"] = poses[i, 0].tolist()
            smplx_param["body_pose"] = poses[i, 1:22].tolist()
            smplx_param["jaw_pose"] = poses[i, 22].tolist()
            smplx_param["leye_pose"] = [0.0, 0.0, 0.0]
            smplx_param["reye_pose"] = [0.0, 0.0, 0.0]
            smplx_param["lhand_pose"] = poses[i, 25:40].tolist()
            smplx_param["rhand_pose"] = poses[i, 40:55].tolist()

            smplx_param["trans"] = transl[i].tolist()
            smplx_param["focal"] = [float(K[0, 0]), float(K[1, 1])]
            smplx_param["princpt"] = [float(K[0, 2]), float(K[1, 2])]
            smplx_param["img_size_wh"] = [img_wh[0], img_wh[1]]
            smplx_param["pad_ratio"] = self.pad_ratio
            with open(os.path.join(out_path, f"{(i+1):05}.json"), "w") as fp:
                json.dump(smplx_param, fp)

    def __call__(self, iamge_list, all_masks,output_path,fps ):
        #start = time.time()
        # all_frames, raw_H, raw_W, fps, offset_w, offset_h, fname_list = load_video(
        #     video_path, pad_ratio=self.pad_ratio
        # )
        all_frames, raw_H, raw_W,  offset_w, offset_h, fname_list = pre_pil_mask(
            iamge_list, pad_ratio=self.pad_ratio, all_masks=all_masks
        )
        # for i,img in enumerate(all_frames):
        #     float_rgb_image = Image.fromarray(img)
        #     float_rgb_image.save(f"f{i}.jpg")


        raw_H, raw_W = all_frames[0].shape[:2]

        self.fps = fps #外置视频的fps
        video_length = len(all_frames)

        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=self.fov, p_x=None, p_y=None, device=self.device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2

        bboxes, frame_ids, frames = self.track(all_frames)
        bboxes, keypoints = self.detect_keypoint2d(bboxes, frames)
        gc.collect()
        torch.cuda.empty_cache()

        poses, betas, transl, verts = self.estimate_pose(
            frame_ids, frames, keypoints, bboxes, raw_K, video_length
        )

        # output_folder = os.path.join(
        #     output_path, video_path.split("/")[-1].split(".")[0]
        # )
        output_folder = output_path
        os.makedirs(output_folder, exist_ok=True)

        all_SMPLX_mesh_frames = self.render_SMPLX_frame(all_frames, verts, raw_K)

        # print(all_SMPLX_mesh_frames.shape)

        smplx_mesh_pils = [Image.fromarray(x) for x in all_SMPLX_mesh_frames]

        # 保存处理视频，开源下次调用
        save_video(smplx_mesh_pils, os.path.join(output_path,"smplx_video.mp4"), fps=fps, quality=7)

        if self.visualize:
            self.save_video(
                all_frames, frame_ids, bboxes, keypoints, verts, raw_K, output_folder
            )

        smplx_output_folder = os.path.join(output_folder, "smplx_params")
        os.makedirs(smplx_output_folder, exist_ok=True)
        self.save_results(
            smplx_output_folder, frame_ids, poses, betas, transl, raw_K, (raw_W, raw_H)
        )
        #duration = time.time() - start
        #print(f"{video_path} processing completed, duration: {duration:.2f}s")
        return smplx_mesh_pils,smplx_output_folder




def save_video(pils, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for pil in pils:
        frame = np.array(pil)
        writer.append_data(frame)
    writer.close()


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, target_camera_list):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if target_camera_list is not None:
                    if os.path.basename(os.path.dirname(path)) in target_camera_list:
                        images.append(path)
                else:
                    images.append(path)

    return images



def get_smplx_mesh(model_path,image_list,mask_list,smplx_mesh_output_path,fps):
    # assert (
    #     torch.cuda.is_available()
    # ), "CUDA is not available, please check your environment"

    args_dict={
        "root": "",
        "save_root": "",
        "save_mesh_root": "",
        "model_path": model_path,
        "pad_ratio": 0.,
        "kp_mode": "vitpose",
        "visualize": False,
        }
    opt = OmegaConf.create(args_dict)


    FOV = 60  # follow the setting of multihmr
    device = torch.device("cuda")

    pipeline = Video2MotionPipeline(
        opt.model_path,
        device,
        kp_mode=opt.kp_mode,
        visualize=opt.visualize,
        pad_ratio=0.,
        fov=FOV,
        fps=fps,
    )


    smplx_mesh_pils,smplx_path=pipeline(image_list, mask_list, smplx_mesh_output_path,fps)
    return smplx_mesh_pils,smplx_path
    # assert False
