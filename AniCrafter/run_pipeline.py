import os
import torch
import imageio
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import json
import math
import cv2

from datetime import datetime
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply
from torchvision.transforms import v2
import folder_paths
from .graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix_refine
from .lhm_runner import HumanLRMInferrer
from .LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel
#from .diffsynth import ModelManager, WanMovieCrafterCombineVideoPipeline
from .diffsynth import ModelManager_ as ModelManager
from .diffsynth import WanMovieCrafterCombineVideoPipeline_ as WanMovieCrafterCombineVideoPipeline


def pad_image_to_aspect_ratio(image, target_width, target_height, background_color=(255, 255, 255)):

    target_ratio = target_width / target_height
    image_ratio = image.width / image.height
    
    if image.width > target_width or image.height > target_height:
        if image_ratio > target_ratio:
            scale_factor = target_width / image.width
        else:
            scale_factor = target_height / image.height
        
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.LANCZOS)
    
    padded_image = ImageOps.pad(
        image, 
        (target_width, target_height), 
        color=background_color,
        centering=(0.5, 0.5)
    )
    
    return padded_image

def crop_image(image_list, max_frames=81):
    return image_list[:max_frames]

def save_video(ref_frame_pils, smplx_pils, blend_pils, video, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for ref_frame, smplx, blend, frame in tqdm(zip(ref_frame_pils, smplx_pils, blend_pils, video), desc="Saving video"):
        h, w, c = np.array(ref_frame).shape
        if h >= w:
            all_frame = np.hstack([
                np.array(ref_frame), 
                np.array(smplx), 
                np.array(blend), 
                np.array(frame)
            ])
        else:
            all_frame = np.vstack([
                np.hstack([
                    np.array(ref_frame), 
                    np.array(smplx), 
                ]), 
                np.hstack([
                    np.array(blend), 
                    np.array(frame)
                ])
            ])
        writer.append_data(all_frame)
    writer.close()



def to_cuda_and_squeeze(value):
    if isinstance(value, dict):  # 如果是字典，则递归处理
        return {k: to_cuda_and_squeeze(v) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):  # 如果是Tensor，则转移到CUDA并压缩
        return value.cuda().squeeze(0)
    return value  # 其他类型的值不处理，直接返回


def PILtoTorch(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

    
def load_camera(pose):
    intrinsic = torch.eye(3)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    image_width, image_height = pose["img_size_wh"]

    c2w = torch.eye(4)
    c2w = c2w.float()

    return c2w.cpu(), intrinsic.cpu(), image_height, image_width # add .cpu()


def video_to_pil_images(video_path, height, width,max_frames=81):
    if video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        pil_images = []
        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break  # 视频结束或读取失败
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
        cap.release()
    elif os.path.isdir(video_path):
        frame_files = sorted([os.path.join(video_path, x) for x in os.listdir(video_path) if x.endswith('.jpg')])
        pil_images = []
        for frame in frame_files:
            frame = cv2.imread(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
    else:
        raise ValueError("Unsupported video format. Please provide a .mp4 file or a directory of images.")
    return pil_images[:max_frames] 


def animate_gs_model(
    offset_xyz, shs, opacity, scaling, rotation, query_points, smplx_data, SMPLX_MODEL
):
    """
    query_points: [N, 3]
    """

    device = offset_xyz.device

    # build cano_dependent_pose
    cano_smplx_data_keys = [
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "expr",
        "trans",
    ]

    merge_smplx_data = dict()
    for cano_smplx_data_key in cano_smplx_data_keys:
        warp_data = smplx_data[cano_smplx_data_key]
        cano_pose = torch.zeros_like(warp_data[:1])

        if cano_smplx_data_key == "body_pose":
            # A-posed
            cano_pose[0, 15, -1] = -math.pi / 6
            cano_pose[0, 16, -1] = +math.pi / 6

        merge_pose = torch.cat([warp_data, cano_pose], dim=0)
        merge_smplx_data[cano_smplx_data_key] = merge_pose

    merge_smplx_data["betas"] = smplx_data["betas"]
    merge_smplx_data["transform_mat_neutral_pose"] = smplx_data[
        "transform_mat_neutral_pose"
    ]

    with torch.autocast(device_type=device.type, dtype=torch.float32):
        mean_3d = (
            query_points + offset_xyz
        )  # [N, 3]  # canonical space offset.

        # matrix to warp predefined pose to zero-pose
        transform_mat_neutral_pose = merge_smplx_data[
            "transform_mat_neutral_pose"
        ]  # [55, 4, 4]
        num_view = merge_smplx_data["body_pose"].shape[0]  # [Nv, 21, 3]
        mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
        query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)
        transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
            num_view, 1, 1, 1
        )

        mean_3d, transform_matrix = (
            SMPLX_MODEL.transform_to_posed_verts_from_neutral_pose(
                mean_3d,
                merge_smplx_data,
                query_points,
                transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                device=device,
            )
        )  # [B, N, 3]

        # rotation appearance from canonical space to view_posed
        num_view, N, _, _ = transform_matrix.shape
        transform_rotation = transform_matrix[:, :, :3, :3]

        rigid_rotation_matrix = torch.nn.functional.normalize(
            matrix_to_quaternion(transform_rotation), dim=-1
        )
        I = matrix_to_quaternion(torch.eye(3)).to(device)

        # inference constrain
        is_constrain_body = SMPLX_MODEL.is_constrain_body
        rigid_rotation_matrix[:, is_constrain_body] = I
        scaling[is_constrain_body] = scaling[
            is_constrain_body
        ].clamp(max=0.02)

        rotation_neutral_pose = rotation.unsqueeze(0).repeat(num_view, 1, 1)

        # QUATERNION MULTIPLY
        rotation_pose_verts = quaternion_multiply(
            rigid_rotation_matrix, rotation_neutral_pose
        )
    
    gaussian_xyz = mean_3d[0]
    canonical_xyz = mean_3d[1]
    gaussian_opacity = opacity
    gaussian_rotation = rotation_pose_verts[0]
    canonical_rotation = rotation_pose_verts[1]
    gaussian_scaling = scaling
    gaussian_rgb = shs

    return gaussian_xyz, canonical_xyz, gaussian_rgb, gaussian_opacity, gaussian_rotation, canonical_rotation, gaussian_scaling, rigid_rotation_matrix


def get_camera_smplx_data(smplx_path,esti_shape):
    with open(smplx_path) as f:
        smplx_raw_data = json.load(f)
    
    smplx_param = {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }
    
    c2w, K, image_height, image_width = load_camera(smplx_param)
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    
    focalX = K[0, 0]
    focalY = K[1, 1]
    FovX = focal2fov(focalX, image_width)
    FovY = focal2fov(focalY, image_height)

    zfar = 1000
    znear = 0.001
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cpu()
    projection_matrix = getProjectionMatrix_refine(torch.Tensor(K).cpu(), image_height, image_width, znear, zfar).transpose(0, 1) #
    #projection_matrix = getProjectionMatrix_refine(K, image_height, image_width, znear, zfar).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    smplx_param['betas'] = esti_shape
    smplx_param['expr'] = torch.zeros((100))

    return {
        'smplx_param': smplx_param, 
        'w2c': w2c, 
        'R': R, 
        'T': T, 
        'K': K, 
        'FoVx': FovX, 
        'FoVy': FovY, 
        'zfar': zfar, 
        'znear': znear, 
        'trans': trans, 
        'scale': scale, 
        'world_view_transform': world_view_transform, 
        'projection_matrix': projection_matrix, 
        'full_proj_transform': full_proj_transform, 
        'camera_center': camera_center, 
    }


def prepare_smplx_model(base_ckpt_path, ):
    SMPLX_MODEL = SMPLXVoxelMeshModel(
        os.path.join(base_ckpt_path,'pretrained_models/human_model_files'),
        gender="neutral",
        subdivide_num=1,
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    ).cuda()

    return SMPLX_MODEL



def prepare_models(dit_path,vae_path, lora_ckpt_path,lora_alpha=1.0,lora_path=None,lora_extra_alpha=None):

    # Load models
    model_manager = ModelManager(device="cpu")
    # model_manager.load_models(
    #     [clip_vision_path],
    #     torch_dtype=torch.float32, # Image Encoder is loaded with float32
    # )
    # model_manager.load_models([dit_path,T5_path,vae_path,],
    #     torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    # )
    model_manager.load_models([dit_path,vae_path,],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )

    model_manager.load_lora_v2_combine([
        os.path.join(lora_ckpt_path, "model-00010-of-00011.safetensors"),
        os.path.join(lora_ckpt_path, "model-00011-of-00011.safetensors"),
        ], lora_alpha=lora_alpha)

    
    if lora_path and lora_extra_alpha :
        for i,lora_p in enumerate (lora_path):
            model_manager.load_lora_v2_combine(lora_p, lora_alpha=lora_extra_alpha[i])
    
    # assert False
   

    pipe = WanMovieCrafterCombineVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda",
    )
    pipe.enable_vram_management()


    return pipe



def predata_for_anicrafter_dispre(frame_process_norm,image_list,character_image,AniCrafter_weigths_path,gaussian_files,clear_cache,preprocess_input,pre_video_dir,use_input_mask,use_bkgd_video,fps,camera_fov,max_frames= 81):

    W, H = image_list[0].size
    H, W = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
    file_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir=os.path.join(folder_paths.models_dir, 'AniCrafter')
    scene_path = os.path.join(folder_paths.base_path, 'custom_nodes/ComfyUI_AniCrafter/AniCrafter/demo/videos/scene_000000')

    pose_estimate_path= os.path.join(base_dir,"pretrained_models/human_model_files")
   # smplx_mesh_pils_origin_ = None
    if pre_video_dir!="none":
        print('use test per video data for preprocessing')
        smplx_mesh_pils_origin = video_to_pil_images(os.path.join(pre_video_dir, 'smplx_video.mp4'), H, W,max_frames)
        smplx_path = os.path.join(pre_video_dir, 'smplx_params')
        if use_bkgd_video is not None:
            print("Using input bkgd video") 
            bkgd_pils_origin=crop_image(use_bkgd_video, max_frames)
        else:
            try:
                bkgd_pils_origin = video_to_pil_images(os.path.join(pre_video_dir, 'bkgd_video.mp4'), H, W,max_frames)
            except:
                raise(f"No bkgd video found in the pre_video_dir{os.path.join(pre_video_dir, 'bkgd_video.mp4')}") 
            

    else:
        if not  preprocess_input : # use test video
            print('use test default data for preprocessing')
            smplx_mesh_pils_origin = video_to_pil_images(os.path.join(scene_path, 'smplx_video.mp4'), H, W,max_frames)
            smplx_path = os.path.join(scene_path, 'smplx_params')
            bkgd_pils_origin = video_to_pil_images(os.path.join(scene_path, 'bkgd_video.mp4'), H, W,max_frames)

        else:
            # 分别获取视频的人物mask，背景内绘，smplx的json参数及合成后的视频 
            
            from .engine.pose_estimation.video2motion import get_smplx_mesh

            print("Start preprocess_input")
            smplx_output_folder=os.path.join(folder_paths.get_output_directory(), f"AniCrafter_{file_prefix}") #保存处理结果，便于下次调用，json目录，smplx_params，smplx_video.mp4,bkgd_video.mp4
            os.makedirs(smplx_output_folder, exist_ok=True)
            mask_save_path=os.path.join(smplx_output_folder, "mask_video.mp4")

            if use_input_mask:
                print("Using input mask video")   
                mask_list=  use_input_mask
            else:
                from .parse_video import get_hunman_mask
                config_path=os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI_AniCrafter/AniCrafter/engine/SegmentAPI/configs/sam2.1_hiera_l.yaml") 
                mask_list=   get_hunman_mask(image_list,config_path,mask_save_path,fps) # 
            print("get mask done")
            
            # get get_smplx_mesh
            print("starting smplx generation")
            smplx_mesh_pils_origin_,smplx_path=get_smplx_mesh(pose_estimate_path,image_list,mask_list,smplx_output_folder,fps,camera_fov)
            print("get splx done")

            smplx_mesh_pils_origin = crop_image(smplx_mesh_pils_origin_,max_frames)

            save_path=os.path.join(smplx_output_folder, 'bkgd_video.mp4')
            pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
            if use_bkgd_video is not None:
                print("Using input bkgd video") 
                bkgd_video_list= use_bkgd_video
            else:
                from .ProPainter.inference_propainter import InferencePropainter
                bkgd_video_list=InferencePropainter(pretrain_model_url,image_list,mask_list,mask_dilation= 16,ref_stride= 2,neighbor_length= 3,resize_ratio= 0.5,model_dir=os.path.join(folder_paths.models_dir, 'AniCrafter/pretrained_models/propainter'),save_path=save_path,save_fps=fps)

            bkgd_pils_origin = crop_image(bkgd_video_list, max_frames)
           
            print('bkgd_pils is done')

   

    character_image_path = os.path.join(folder_paths.get_input_directory(), f"AniCrafter_{file_prefix}.jpg")
    character_image.save(character_image_path)
    save_gaussian_path = character_image_path.replace('.jpg', '_gaussian.pth')
    #save_video_path = os.path.join(save_root, f'{os.path.basename(scene_path)}/{os.path.basename(character_image_path).split(".")[0]}.mp4')
    os.makedirs(os.path.dirname(save_gaussian_path), exist_ok=True)

    HumanLRMInferrer_ = HumanLRMInferrer(base_dir)
    HumanLRMInferrer_.load_model()

    print('lhm_runner init')
    gaussians_list, body_rgb_pil, crop_body_pil = HumanLRMInferrer_.infer(character_image_path, save_gaussian_path,gaussian_files,clear_cache) #TODO check reload
    print('lhm_runner infer done')
   
    body_rgb_pil_pad = pad_image_to_aspect_ratio(crop_body_pil, W, H)
    dxdydz, xyz, rgb, opacity, scaling, rotation, transform_mat_neutral_pose, esti_shape, body_ratio, have_face = gaussians_list

    
    smplx_json_paths = sorted(os.path.join(smplx_path, x) for x in os.listdir(smplx_path))[:max_frames]
    smplx_mesh_tensors = [torch.from_numpy(np.array(smplx_mesh_pil)) / 255. for smplx_mesh_pil in smplx_mesh_pils_origin]


    smplx_mask_nps = []
    bkgd_nps = []
    for smplx_mesh_tensor, bkgd_pil in zip(smplx_mesh_tensors, bkgd_pils_origin):
        smplx_mask = (smplx_mesh_tensor <= 0.01).all(dim=-1, keepdim=False).float()  # [720, 1280]
        smplx_mask_np = np.uint8(255 - smplx_mask.detach().cpu().numpy() * 255)  # [80, h, w]
        smplx_mask_nps.append(smplx_mask_np)
        bkgd_nps.append(np.array(bkgd_pil))


    SMPLX_MODEL=prepare_smplx_model(AniCrafter_weigths_path)
    print('SMPLX_MODEL init')
    
    blend_pils_origin = []
    for bkgd_pil, smplx_json_path in tqdm(zip(bkgd_pils_origin, smplx_json_paths), desc="Rendering Avatar", total=len(bkgd_pils_origin)):

        batch = {
            key: to_cuda_and_squeeze(value) 
            for key, value in get_camera_smplx_data(
                smplx_json_path,esti_shape
            ).items()
        }

        render_image_width, render_image_height = int(batch['smplx_param']["img_size_wh"][0]), int(batch['smplx_param']["img_size_wh"][1])

        gaussian_canon_dxdydz = dxdydz.cuda()
        query_points = xyz.cuda()
        gaussian_canon_rgb = rgb.cuda()
        gaussian_canon_opacity = opacity.cuda()
        gaussian_canon_scaling = scaling.cuda()
        gaussian_canon_rotation = rotation.cuda()
        transform_mat_neutral_pose = transform_mat_neutral_pose.cuda()
        esti_shape = esti_shape.cuda()

        smplx_data = {
            'betas': batch['smplx_param']['betas'].unsqueeze(0), 
            'root_pose': batch['smplx_param']['root_pose'].unsqueeze(0), 
            'body_pose': batch['smplx_param']['body_pose'].unsqueeze(0), 
            'jaw_pose': batch['smplx_param']['jaw_pose'].unsqueeze(0), 
            'leye_pose': batch['smplx_param']['leye_pose'].unsqueeze(0), 
            'reye_pose': batch['smplx_param']['reye_pose'].unsqueeze(0), 
            'lhand_pose': batch['smplx_param']['lhand_pose'].unsqueeze(0), 
            'rhand_pose': batch['smplx_param']['rhand_pose'].unsqueeze(0), 
            'trans': batch['smplx_param']['trans'].unsqueeze(0), 
            'expr': batch['smplx_param']['expr'].unsqueeze(0), 
            'transform_mat_neutral_pose': transform_mat_neutral_pose, 
        }

         

        gaussian_xyz, canonical_xyz, gaussian_rgb, gaussian_opacity, gaussian_rotation, canonical_rotation, gaussian_scaling, transform_matrix = \
            animate_gs_model(
                gaussian_canon_dxdydz, gaussian_canon_rgb, gaussian_canon_opacity, 
                gaussian_canon_scaling, gaussian_canon_rotation, query_points, 
                smplx_data, SMPLX_MODEL
            )
        
        # Set up rasterization configuration
        tanfovx = math.tan(batch['FoVx'] * 0.5)
        tanfovy = math.tan(batch['FoVy'] * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=render_image_height,
            image_width=render_image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=1.,
            viewmatrix=batch['world_view_transform'],
            projmatrix=batch['full_proj_transform'],
            sh_degree=0,
            campos=batch['camera_center'],
            prefiltered=False,
            debug=False,
          
        )
            
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth, alpha = rasterizer(
            means3D = gaussian_xyz, 
            means2D = torch.zeros_like(canonical_xyz, dtype=canonical_xyz.dtype, requires_grad=False, device="cuda") + 0, 
            shs = None, 
            colors_precomp = gaussian_rgb, 
            opacities = gaussian_opacity, 
            scales = gaussian_scaling, 
            rotations = gaussian_rotation, 
            cov3D_precomp = None
        )
     

        blend_image = rendered_image * alpha + PILtoTorch(bkgd_pil.resize((render_image_width, render_image_height), Image.Resampling.LANCZOS)).cuda() * (1 - alpha)

        blend_image = Image.fromarray(np.uint8(blend_image.permute(1, 2, 0).detach().cpu().numpy() * 255))
        blend_image = blend_image.resize((W, H), Image.Resampling.LANCZOS)

        blend_pils_origin.append(blend_image)
    # if  smplx_mesh_pils_origin_ is None:
    #     smplx_mesh_pils_origin = video_to_pil_images(os.path.join(scene_path, 'smplx_video.mp4'), H, W,max_frames)
    # else:
    #     smplx_mesh_pils_origin = crop_image(smplx_mesh_pils_origin_,max_frames)#TODO check if need maxframe
    ref_frame = body_rgb_pil_pad
    ref_frame_tensor = frame_process_norm(ref_frame).cuda()

    #ref_frame_pils_origin = [ref_frame for _ in range(max_frames)]

    blend_tensor = torch.stack([frame_process_norm(ss) for ss in blend_pils_origin], dim=0).cuda().permute(1, 0, 2, 3)
    smplx_tensor = torch.stack([frame_process_norm(ss) for ss in smplx_mesh_pils_origin], dim=0).cuda().permute(1, 0, 2, 3)

    ref_combine_blend_tensor = torch.cat([ref_frame_tensor.unsqueeze(1), blend_tensor[:, :-1]], dim=1)
    ref_combine_smplx_tensor = torch.cat([ref_frame_tensor.unsqueeze(1), smplx_tensor[:, :-1]], dim=1)
    #print(ref_combine_blend_tensor.shape, ref_combine_smplx_tensor.shape,ref_combine_blend_tensor.is_cuda, ref_combine_smplx_tensor.is_cuda,ref_combine_blend_tensor.dtype) #torch.Size([3, 81, 384, 384]) torch.Size([3, 81, 384, 384])
    return ref_combine_blend_tensor,ref_combine_smplx_tensor,H, W




def infer_anicrafter(pipe, ref_combine_blend_tensor,ref_combine_smplx_tensor,height,width,num_inference_steps,seed,use_teacache,cfg_value,use_tiled,text_emb,image_emb,wan_repo="Wan2.1-I2V-14B-720P"):
    
  
    
    # Image-to-video
    try: 
        video = pipe(
            prompt=None ,
            negative_prompt=None,
            ref_combine_blend_tensor=ref_combine_blend_tensor, 
            ref_combine_smplx_tensor=ref_combine_smplx_tensor, 
            input_image=None,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_value, 
            seed=seed, 
            tiled=use_tiled, 
            height=height,
            width=width,
            num_frames=image_emb.get("max_frames"),
            tea_cache_l1_thresh=0.3 if use_teacache else None,
            tea_cache_model_id=wan_repo if use_teacache else None,
            prompt_emb_posi=text_emb.get("prompt_emb_posi"),
            prompt_emb_nega=text_emb.get("prompt_emb_nega") if cfg_value!=1 else None,
            clip_context=image_emb.get("clip_context"),

        )
        
    except Exception as e:
        print(e)
        pipe.to("cpu")
        return [Image.new('RGB', (width, height), (255, 255, 255)),]*image_emb.get("max_frames")

    return video



