import numpy as np
import os
import cv2
import torch
import argparse
from PIL import Image
import imageio
from tqdm import tqdm
import torch.distributed as dist

from .engine.SegmentAPI.SAM import SAM2Seg



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



def save_video(pils, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in pils:
        all_frame = np.array(frame)
        writer.append_data(all_frame)
    writer.close()




def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def get_hunman_mask(image_list,):

    
    parsingnet = SAM2Seg()

    masks = []
    
    for frame in image_list:
        parsing_out = parsingnet._forward(np.array(frame), bbox=None)
        alpha = Image.fromarray((parsing_out.masks * 255).astype(np.uint8))
        masks.append(alpha)

    #save_video(masks, save_path, fps=15, quality=9)
    return masks
    # assert False