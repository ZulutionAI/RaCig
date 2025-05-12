import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionXLPipeline
from PIL import Image
from typing import Dict, List, Optional
import os
import re
import json
import numpy as np
import warnings
import time
import torch.nn.functional as F
from ip_adapter.ip_adapter_mask import IPAdapterPlusXL_RP, IPAdapterPlusXL_RP_MultiRef, IPAdapterPlusXL_RP_MultiRef_Double_Adapter
from ip_adapter.utils import register_cross_attention_hook
from .retrieval_match import Retriever, Matcher_Grounding_DINO
from .utils import get_center, assign_by_nearest, draw_bodypose_openpose, resize_pose, resize_segmap, expand_bbox, draw_mask_fine, draw_face_mask, pil_images_to_matplotlib_figure
from .character import Character
from .scheduler import DPMSolverPlusSDEScheduler
import random 
import cv2
import scipy
from safetensors import safe_open
from typing import List, Optional
from .custom_pipeline import StableDiffusionXLCustomControlNetPipeline

def set_all_seed(seed=42):
    # Setting seed for Python's built-in random module
    random.seed(seed)
    # Setting seed for NumPy
    np.random.seed(seed)
    # Setting seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensuring reproducibility for convolutional layers in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RaCigPipeline:
    def __init__(
            self,
            retrieval_model_name,
            index_path,
            json_dict_path,
            data_root,
            base_model_path = "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder_path = "models/image_encoder",
            ip_ckpt = "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
            ip_ckpt_body = None,
            controlnet_path = "xinsir/controlnet-openpose-sdxl-1.0",
            height=1344,
            width=768,
            device = "cuda",
            multi_ref_method = 'average',
            seperate_ref = True
            ):
        
        self.height = height
        self.width = width
        self.device = device
        self.multi_ref_method = multi_ref_method
        self.seperate_ref = seperate_ref

        if seperate_ref: #如果要使用脸部和衣服分开ref，则必须使用stack模型得到img embedding
            assert multi_ref_method == 'stack'

        if ip_ckpt_body is not None: #如果要使用两个ip-adapter（脸部使用adapter-face，身体使用adapter-body）
            assert multi_ref_method == 'stack' and seperate_ref
        
        # 初始化retriever
        self.retriever =  Retriever(
            clip_model_name=retrieval_model_name,
            index_path=index_path,
            data_root=data_root,
            json_dict_path=json_dict_path,
            device=self.device
        )
        
        # 初始化Matcher
        self.matcher_gd = Matcher_Grounding_DINO()

        controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0", use_safetensors=True, torch_dtype=torch.float16).to(device)
        if os.path.splitext(controlnet_path)[-1] == ".safetensors":
            state_dict = {"controlnet": {}}
            with safe_open(controlnet_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("controlnet."):
                        state_dict["controlnet"][key.replace("controlnet.", "")] = f.get_tensor(key)
            controlnet.load_state_dict(state_dict["controlnet"],strict=True)

        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            "models/sdxl/dreamshaper.safetensors",
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)

        pipe.scheduler = DPMSolverPlusSDEScheduler()
        if multi_ref_method is not None:
            if ip_ckpt_body is None:
                self.ip_model = IPAdapterPlusXL_RP_MultiRef(
                    pipe, 
                    image_encoder_path, 
                    ip_ckpt, 
                    device, 
                    num_tokens=16,
                    height=height,
                    width=width,
                    ref_method=multi_ref_method
                    ) 
            else:
                self.ip_model = IPAdapterPlusXL_RP_MultiRef_Double_Adapter(
                    pipe, 
                    image_encoder_path, 
                    ip_ckpt, 
                    ip_ckpt_body,
                    device, 
                    num_tokens=16,
                    height=height,
                    width=width,
                    ref_method=multi_ref_method
                    ) 
        else:
            self.ip_model = IPAdapterPlusXL_RP(
                pipe, 
                image_encoder_path, 
                ip_ckpt, 
                device, 
                num_tokens=16,
                height=height,
                width=width,
                )
        
    def __call__(
            self,
            character_list: List[Character],
            prompt: List[str],
            bg_prompt: str = None,
            bg_img: Image = None,
            num_samples: int = 4,
            top_k: int = 10,
            skeleton_id: int=-1,
            sample_method = 'random',
            num_inference_steps = 13,
            seed=42, 
            guidance_scale=2.5,
            kp_conf=0.1,
            ref_scale=1.0
            ):
        
        """
        Args:
            prompt: Regional Prompter Format: common prompt \n prompt1 \n prompt2
            e.g. 2characters, Laura kissing Nacho, afternoon, wide shot,\u00a0(2d artistic sketch:1.5), flat color, line art, 
            low saturation, (thin outlines:1.5), trending on Pixiv, (low contrast:1.5), 
            (2d style:1.5), delicate face
            \nLaura, (1Female:1.5), Sales Director, 30s, Caucasian, Auburn Hair, Blazer
            \nNacho, (1Male:1.5), wealthy CEO, European, fit, short dark, suit
        """

        set_all_seed(seed)
        start_time_total = time.time()
        # 1. Parse prompt
        ori_prompts = prompt.lower().split('\n')
        if len(ori_prompts) == 4:
            common_prompt = ori_prompts[0].lower() + ori_prompts[1].lower() #如果有四句话，common prompt是前两句
        else:
            common_prompt = ori_prompts[0].lower()
        
        seperate_prompts = ori_prompts[-2:]
        # 1.1 Convert common prompt for retrieval
        # 把common prompt中的名字替换成character的性别
        character_name_gender = {character.name.lower(): character.gender for character in character_list}
        common_prompt_retrieval = common_prompt
        common_prompt_retrieval = ','.join(common_prompt_retrieval.split(',')[:3])

        # 1.2 Conver common prompt for character
        # 对于每个character的prompt，把common prompt中的名字都替换成某个character的性别，保证Regional prompter生图的一致性
        character_name = [character.name for character in character_list]
        pattern = re.compile("|".join(re.escape(name) for name in character_name))
        for character_id, character in enumerate(character_list):
            character.converted_prompt = pattern.sub(character.gender, common_prompt)
            # character.converted_prompt = common_prompt
                        
        # 2. Retrieve
        assert sample_method in ['random', 'index']

        start_time = time.time()

      
        # 以下保证检索出有效的skeleton和img：保证skeleton的人物数量和character数量一致
        success = False
        for attempt in range(10):
            now_top_k = top_k + attempt * 5
            retrieved = self.retriever.search(common_prompt_retrieval, now_top_k)
            traversed = set()
            idx = skeleton_id
            draw_pose_error = False
            while True:
                if sample_method == 'random':
                    idx = random.randint(0, now_top_k-1)
                elif sample_method == 'index':
                    idx += 1
                else:
                    raise ValueError
                traversed.add(idx)
                if len(traversed) == now_top_k:
                    success = False
                    break

                print(f'Using skeleton: {idx}')
                skeleton_path = retrieved['skeleton_paths'][idx]
                img_path = retrieved['img_paths'][idx]
                try:
                    img = Image.open(img_path)
                except Exception as e: 
                    print(f"{e}")
                    continue
                
                json_path = retrieved['json_paths'][idx]
                seg_path = retrieved['seg_paths'][idx]
                face_seg_path = retrieved['face_seg_paths'][idx]
                head_seg_path = retrieved['head_seg_paths'][idx]
                try:
                    with open(json_path) as f:
                        info_dict = json.load(f)
                except Exception as e: 
                    print(f"{e}")
                    continue
                
                segments = info_dict['segments']
                segments = [segment for segment in segments if segment['is_clear'] == 'yes']

                # Read head and face segmap: assert id must in these two maps
                try:
                    if '.npz' in face_seg_path and '.npz' in head_seg_path:
                        loaded_sparse_mask = scipy.sparse.load_npz(face_seg_path)
                        # 将稀疏矩阵转换回正常的numpy数组
                        face_seg_map = loaded_sparse_mask.toarray()
                        loaded_sparse_mask = scipy.sparse.load_npz(head_seg_path)
                        # 将稀疏矩阵转换回正常的numpy数组
                        head_seg_map = loaded_sparse_mask.toarray()
                    else:
                        face_seg_map = np.load(face_seg_path)
                        head_seg_map = np.load(head_seg_path)
                except Exception as e: 
                    print(f"{e}")
                    continue
                
                all_in_face_head_seg_map = True
                for segment in segments:
                    if (segment['id'] not in np.unique(face_seg_map)) or (segment['id'] not in np.unique(head_seg_map)):
                        all_in_face_head_seg_map = False
                if not all_in_face_head_seg_map:
                    print("Face num != Segments num")
                    continue

                with open(skeleton_path) as f:
                    skeleton_json = json.load(f)

                # skeleton_json = [[skeleton_json[0][0]]] #debug

                skeletons = [skeleton['skeleton'] for skeleton in skeleton_json if skeleton['skeleton'] is not None]
                len_ske = len(skeletons)
                len_seg = len(segments)

                if (len_seg != len(character_list)) or (len_seg != len_ske):
                    print(f"Skeleton num:{len_ske}, Segments num:{len_seg}, Character num:{len(character_list)} are not equal")
                    continue

                seg_map = np.load(seg_path)
                H, W = seg_map.shape
                pose_img = np.zeros((*seg_map.shape,3))

                # Draw pose
                for person in skeletons:

                    keypoints = np.zeros((18,3))
                    i = 0
                    for point, score in zip(person['keypoints'], person['keypoint_scores']):
                            normalized_x = point[0] / W
                            normalized_y = point[1] / H
                            keypoints[i] = [normalized_x, normalized_y, score]
                            i += 1
                    neck = (keypoints[5,:] + keypoints[6,:])/2
                    keypoints[i] = neck
                    # transform mmpose keypoints to openpose keypoints
                    openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
                    mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
                    new_keypoints = keypoints[:, :]
                    new_keypoints[openpose_idx, :] = keypoints[mmpose_idx, :]
                    # draw skeleton of current person
                    try:
                        pose_img = draw_bodypose_openpose(pose_img, new_keypoints, kp_conf)
                    except Exception as e:
                        draw_pose_error = True
                    
                if draw_pose_error:
                    print(f"Drawing pose error, continue to next one: {e}")
                    continue
                success = True
                break
            if success:
                break
        if attempt == 9:
            raise ValueError("检索失败:尝试10次后仍未找到合适的骨架和分割图像")
            
        end_time = time.time()
        retrieval_time = end_time - start_time

        # 3. Match
        start_time = time.time()
        
        match_dict = {}
        if len(character_list) == 1:
            match_dict[character_list[0].name] = segments[0]['id']

        elif len(character_list) == 2:
            # 按照关系去用grounding dino做匹配
            match_dict = self.matcher_gd.match_subjects(
                sentence=common_prompt.split(',')[1],
                img=img,
                segmap=seg_map,
                segments=segments,
                character_list=character_list
            )

        end_time = time.time()
        match_time = end_time - start_time

        # 4. Generate
        start_time = time.time()

        # 4.1 Resize seg_map and pose img：resize成1344，768
        resized_pose_img, resized_seg_map = resize_pose(pose_img, self.height, self.width), resize_segmap(seg_map, self.height, self.width)
        resize_face_seg_map = resize_segmap(face_seg_map, self.height, self.width)
        resize_head_seg_map = resize_segmap(head_seg_map, self.height, self.width)

        pose_img = Image.fromarray(resized_pose_img)
        seg_map = torch.from_numpy(resized_seg_map)
        face_seg_map = torch.from_numpy(resize_face_seg_map)
        head_seg_map = torch.from_numpy(resize_head_seg_map)

        kernel = torch.ones((1, 1, 30, 30), dtype=torch.float32).to('cuda', torch.float32)

        ref_img_list = []
        prompts = []
        masks = []
        skeletons = {item['id']: item for item in skeleton_json}
        for character in character_list:
            for k in match_dict.keys():
                name = character.name.strip(' ').split(' ')
                name_in = True
                for word in name:
                    if word not in k:
                        name_in = False
                    else:
                         name_in = True
                if name_in:    
                    character.seg_id = match_dict[k]
                    break
            # 此处需要把mask进行膨胀使得其范围更广
            mask = (F.conv2d((seg_map == character.seg_id).unsqueeze(0).unsqueeze(0).float().cuda(), kernel, padding='same') > 0).to(torch.float) # (1, 1, h, w)
            masks.append(mask) # (1, 1, h, w)
            ref_img_list.append(character.ref_img)
            prompts.append(character.converted_prompt)

        # If 2 Characters, fix the intersection of two masks:mask交集区域用最近邻算法进行分配
        if len(masks) == 2:
            mask0, mask1 = masks[0].squeeze(0), masks[1].squeeze(0)
            intersection = (mask0 > 0) & (mask1 > 0)
            mask0[intersection] = 0
            mask1[intersection] = 0
            mask0, mask1 = assign_by_nearest(mask0, mask1, intersection)
            masks = [mask0.unsqueeze(0), mask1.unsqueeze(0)]

        # Option: Sperate face and body 如果脸部和衣服进行分开ref，则需要需要对每个人的mask进一步分成脸部mask和衣服mask
        if self.seperate_ref:
            masks_seperate = []
            img_np = resize_pose(np.array(img), self.height, self.width)
            masks_visualize = []
            for character, mask in zip(character_list, masks):
                face_mask = (F.conv2d((face_seg_map == character.seg_id).unsqueeze(0).unsqueeze(0).cuda().float(), kernel, padding='same') > 0).to(torch.float)
                head_mask = (F.conv2d((head_seg_map == character.seg_id).unsqueeze(0).unsqueeze(0).cuda().float(), kernel, padding='same') > 0).to(torch.float)
                head_face_mask = (face_mask + head_mask) > 0

                mask_head_face = mask * head_face_mask
                mask_body = mask * (~head_face_mask)

                mask_face_vis = img_np * mask_head_face.cpu().numpy()[0,0,...][...,np.newaxis] # （h*w）
                mask_body_vis = img_np * mask_body.cpu().numpy()[0,0,...][...,np.newaxis]
                masks_visualize.append(Image.fromarray(mask_face_vis.astype(np.uint8)))
                masks_visualize.append(Image.fromarray(mask_body_vis.astype(np.uint8)))

                mask = torch.cat([mask_head_face, mask_body], dim=0) # (2, 1, h, w)
                masks_seperate.append(mask)
            masks = masks_seperate

            # if len(masks_visualize) < 5:
            #     for _ in range(5-len(masks_visualize)):
            #         masks_visualize.append(Image.fromarray(np.zeros_like(mask_body_vis, dtype=np.uint8)))

        masks = torch.stack([mask.repeat(1, num_samples*2, 1, 1).to(self.device) for mask in masks])
        
        # When multi_ref 如果每个character人脸有多张ref img，则需要记录具体ref img数量
        if self.multi_ref_method is not None:
            if self.multi_ref_method == 'stack':
                if self.seperate_ref: # Remove body ref 如果人脸和衣服分开ref，这里只记录人脸的数量，排除衣服图片，所以减一
                    multi_ref_num = [len(ref_img) - 1 for ref_img in ref_img_list]
                else:
                    multi_ref_num = [len(ref_img) for ref_img in ref_img_list]

        # Background 不管是否有背景ref img和prompt我们都将其加进来（空字符串或黑图）以便于后续的batch运算
        if bg_prompt is not None:
            prompts.append(bg_prompt)
        else:
            prompts.append('')

        if bg_img is not None:
            if self.multi_ref_method == 'stack':
                ref_img_list.append([bg_img])
            else:
                ref_img_list.append(bg_img) # TODO: Adapt bg img when multi ref
        else:
            if self.multi_ref_method == 'stack':
                ref_img_list.append([Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))])
            else:
                ref_img_list.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        multi_ref_num.append(1) # Add bg ref， 背景也是作为ref img所以需要记录进ref img数量
        mask_prepare_time = time.time() - start_time

        start_time = time.time()
        images_list = self.ip_model.generate(
            pil_image=ref_img_list, 
            prompt=prompts, 
            ori_prompt=common_prompt,
            num_samples=num_samples, 
            num_inference_steps=num_inference_steps, 
            seed=seed, 
            width=self.width, 
            height=self.height,
            image=pose_img, 
            controlnet_conditioning_scale=0.7, 
            guidance_scale=guidance_scale,
            scale=ref_scale,
            cross_attention_kwargs={
                'masks':masks, 
                "use_bg_prompt": bg_prompt is not None,
                "use_bg_img": bg_img is not None,       
                "multi_ref_num": multi_ref_num if self.multi_ref_method is not None and self.multi_ref_method == 'stack' else None      
                })
        end_time = time.time()
        generate_time = end_time-start_time
        total_time = end_time - start_time_total
        debug = True
        # if debug:
        #     images.append(img)
        #     # images.append(Image.fromarray(seg_map.cpu().numpy()))
        #     # for i in range(masks.shape[0]):
        #     #     for j in range(masks.shape[1]):
        #     #         images.append(Image.fromarray((resize_pose(np.array(img), self.height, self.width)*masks[i,j,0,...].cpu().numpy()[...,np.newaxis]).astype(np.uint8)))
        #     images.append(pose_img)
        #     fig = pil_images_to_matplotlib_figure(images)
        #     fig.savefig('generated.png')

        print('retrieval time:', retrieval_time, 'match_time:', match_time, 'mask prepare time:', mask_prepare_time, 'generate_time:', generate_time, 'total_time:', total_time)
        return images_list, img, pose_img, masks_visualize, idx, json_path

class RaCigPipelineForGradio:
    def __init__(
            self,
            skeleton_model_config_path: str,
            retrieval_model_name: str,  
            index_path: str,  
            json_dict_path: str, 
            data_root: str, 
            base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
            image_encoder_path: str = "models/image_encoder",
            ip_ckpt: str = "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
            ip_ckpt_body: Optional[str] = None,
            controlnet_path: str = "xinsir/controlnet-openpose-sdxl-1.0",
            height: int = 1344,
            width: int = 768,
            device: str = "cuda",
            multi_ref_method: str = 'average',
            seperate_ref: bool = True
            ):
        
        self.height = height
        self.width = width
        self.device = device
        self.multi_ref_method = multi_ref_method
        self.seperate_ref = seperate_ref

        if seperate_ref: #如果要使用脸部和衣服分开ref，则必须使用stack模型得到img embedding
            assert multi_ref_method == 'stack'

        if ip_ckpt_body is not None: #如果要使用两个ip-adapter（脸部使用adapter-face，身体使用adapter-body）
            assert multi_ref_method == 'stack' and seperate_ref
        
        # 初始化retriever
        self.retriever =  Retriever(
            clip_model_name=retrieval_model_name,
            index_path=index_path,
            data_root=data_root,
            json_dict_path=json_dict_path,
            device=self.device
        )
        
        self.matcher_gd = Matcher_Grounding_DINO()

        controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0", use_safetensors=True, torch_dtype=torch.float16).to(device)
        if os.path.splitext(controlnet_path)[-1] == ".safetensors":
            state_dict = {"controlnet": {}}
            with safe_open(controlnet_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("controlnet."):
                        state_dict["controlnet"][key.replace("controlnet.", "")] = f.get_tensor(key)
            controlnet.load_state_dict(state_dict["controlnet"])

        pipe = StableDiffusionXLCustomControlNetPipeline.from_single_file(
            "/ucloud/mnt-20T/huangyuqiu/models/sdxl/Stable-diffusion/dreamshaper.safetensors",
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)
        pipe.unet = register_cross_attention_hook(pipe.unet)

        pipe.scheduler = DPMSolverPlusSDEScheduler()

        
        if multi_ref_method is not None:
            if ip_ckpt_body is None:
                self.ip_model = IPAdapterPlusXL_RP_MultiRef(
                    pipe, 
                    image_encoder_path, 
                    ip_ckpt, 
                    device, 
                    num_tokens=16,
                    height=height,
                    width=width,
                    ref_method=multi_ref_method
                    ) 
            else:
                self.ip_model = IPAdapterPlusXL_RP_MultiRef_Double_Adapter(
                    pipe, 
                    image_encoder_path, 
                    ip_ckpt, 
                    ip_ckpt_body,
                    device, 
                    num_tokens=16,
                    height=height,
                    width=width,
                    ref_method=multi_ref_method
                    ) 
        else:
            self.ip_model = IPAdapterPlusXL_RP(
                pipe, 
                image_encoder_path, 
                ip_ckpt, 
                device, 
                num_tokens=16,
                height=height,
                width=width,
                )
            
    def retrieval_skeleton_show(self, num_show, character_list, common_prompt_for_geometry, sample_method, top_k, skeleton_id, kp_conf, add_hair=True, out_mask=False):
        # Retrieve
        assert sample_method in ['random', 'index']
        # 以下保证检索出有效的skeleton和img：保证skeleton的人物数量和character数量一致
        success_count = 0
        out_skeletons_list, pose_img_list, match_dict_list, masks_list = [], [], [], []
        
        retrieved = self.retriever.search(common_prompt_for_geometry, top_k)
        traversed = set()
        idx = skeleton_id

        while success_count < num_show and len(traversed) < top_k:
            if sample_method == 'random':
                idx = random.choice([i for i in range(top_k) if i not in traversed])
            elif sample_method == 'index':
                idx = (idx + 1) % top_k
            else:
                raise ValueError
            traversed.add(idx)

            print(f'Using skeleton: {idx}')
            skeleton_path = retrieved['skeleton_paths'][idx]
            img_path = retrieved['img_paths'][idx]
            try:
                img = Image.open(img_path)
            except Exception as e: 
                print(f"{e}")
                continue
            
            json_path = retrieved['json_paths'][idx]
            seg_path = retrieved['seg_paths'][idx]
            face_seg_path = retrieved['face_seg_paths'][idx]
            head_seg_path = retrieved['head_seg_paths'][idx]
            try:
                with open(json_path) as f:
                    info_dict = json.load(f)
            except Exception as e: 
                print(f"{e}")
                continue
            
            segments = info_dict['segments']
            segments = [segment for segment in segments if segment['is_clear'] == 'yes']

            # Read head and face segmap: assert id must in these two maps
            try:
                if '.npz' in face_seg_path and '.npz' in head_seg_path:
                    loaded_sparse_mask = scipy.sparse.load_npz(face_seg_path)
                    face_seg_map = loaded_sparse_mask.toarray()
                    loaded_sparse_mask = scipy.sparse.load_npz(head_seg_path)
                    head_seg_map = loaded_sparse_mask.toarray()
                else:
                    face_seg_map = np.load(face_seg_path)
                    head_seg_map = np.load(head_seg_path)
            except Exception as e: 
                print(f"{e}")
                continue
            
            all_in_face_head_seg_map = True
            for segment in segments:
                if (segment['id'] not in np.unique(face_seg_map)) or (segment['id'] not in np.unique(head_seg_map)):
                    all_in_face_head_seg_map = False
            if not all_in_face_head_seg_map:
                print("Face num != Segments num")
                continue

            with open(skeleton_path) as f:
                skeleton_json = json.load(f)

            len_ske = len([skeleton['skeleton'] for skeleton in skeleton_json if skeleton['skeleton'] is not None])
            len_seg = len(segments)

            if (len_seg != len(character_list)) or (len_seg != len_ske):
                print(f"Skeleton num:{len_ske}, Segments num:{len_seg}, Character num:{len(character_list)} are not equal")
                continue

            seg_map = np.load(seg_path)
            H, W = seg_map.shape
            pose_img = np.zeros((*seg_map.shape,3))

            # Draw pose
            out_skeletons = {}
            draw_pose_error = False
            for person in skeleton_json:
                keypoints = np.zeros((18,3))
                i = 0
                for point, score in zip(person['skeleton']['keypoints'], person['skeleton']['keypoint_scores']):
                        normalized_x = point[0] / W
                        normalized_y = point[1] / H
                        keypoints[i] = [normalized_x, normalized_y, score]
                        i += 1
                neck = (keypoints[5,:] + keypoints[6,:])/2
                keypoints[i] = neck
                # transform mmpose keypoints to openpose keypoints
                openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
                mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
                new_keypoints = keypoints[:, :]
                new_keypoints[openpose_idx, :] = keypoints[mmpose_idx, :]
                out_skeletons[person['id']] = new_keypoints
                # draw skeleton of current person
                try:
                    pose_img = draw_bodypose_openpose(pose_img, new_keypoints, kp_conf)
                except Exception as e:
                    draw_pose_error = True
                    break
            
            if draw_pose_error:
                print(f"Drawing pose error, continue to next one: {e}")
                continue

            # Match
            match_dict = {}
            if len(character_list) == 1:
                match_dict[character_list[0].name] = segments[0]['id']
            elif len(character_list) == 2:
                # 按照关系去用grounding dino做匹配
                match_dict = self.matcher_gd.match_subjects(
                    sentence=common_prompt_for_geometry.split(',')[1],
                    img=img,
                    segmap=seg_map,
                    segments=segments,
                    character_list=character_list
                )

            resized_pose_img = resize_pose(pose_img, self.height, self.width)
            pose_img = Image.fromarray(resized_pose_img)

            out_skeletons_list.append(out_skeletons)
            pose_img_list.append(pose_img)
            match_dict_list.append(match_dict)

            success_count += 1

            if out_mask:
                resized_seg_map = resize_segmap(seg_map, self.height, self.width)
                resize_face_seg_map = resize_segmap(
                    face_seg_map, self.height, self.width)
                resize_head_seg_map = resize_segmap(
                    head_seg_map, self.height, self.width)

                seg_map = torch.from_numpy(resized_seg_map)
                face_seg_map = torch.from_numpy(resize_face_seg_map)
                head_seg_map = torch.from_numpy(resize_head_seg_map)

                kernel = torch.ones((1, 1, 30, 30), dtype=torch.float32).to(
                    'cuda', torch.float32)
                
                masks = []
                for character in character_list:
                    for k in match_dict.keys():
                        name = character.name.strip(' ').split(' ')
                        name_in = True
                        for word in name:
                            if word not in k:
                                name_in = False
                            else:
                                name_in = True
                        if name_in:
                            character.seg_id = match_dict[k]
                            break
                    # 此处需要把mask进行膨胀使得其范围更广
                    mask = (F.conv2d((seg_map == character.seg_id).unsqueeze(0).unsqueeze(
                        0).float().cuda(), kernel, padding='same') > 0).to(torch.float)  # (1, 1, h, w)
                    masks.append(mask)  # (1, 1, h, w)


                # If 2 Characters, fix the intersection of two masks:mask交集区域用最近邻算法进行分配
                if len(masks) == 2:
                    mask0, mask1 = masks[0].squeeze(0), masks[1].squeeze(0)
                    intersection = (mask0 > 0) & (mask1 > 0)
                    mask0[intersection] = 0
                    mask1[intersection] = 0
                    mask0, mask1 = assign_by_nearest(mask0, mask1, intersection)
                    masks = [mask0.unsqueeze(0), mask1.unsqueeze(0)]

                # Option: Sperate face and body 如果脸部和衣服进行分开ref，则需要需要对每个人的mask进一步分成脸部mask和衣服mask
                masks_seperate = []
                for character, mask in zip(character_list, masks):
                    face_mask = (F.conv2d((face_seg_map == character.seg_id).unsqueeze(0).unsqueeze(
                        0).cuda().float(), kernel, padding='same') > 0).to(torch.float)
                    head_mask = (F.conv2d((head_seg_map == character.seg_id).unsqueeze(0).unsqueeze(
                        0).cuda().float(), kernel, padding='same') > 0).to(torch.float)
                    if add_hair:
                        head_face_mask = (face_mask + head_mask) > 0
                    else:
                        head_face_mask = (face_mask) > 0
                    mask_head_face = mask * head_face_mask
                    mask_body = mask * (~head_face_mask)
                    mask = torch.cat([mask_head_face, mask_body], dim=0)  # (2,1,h,w)
                    masks_seperate.append(mask)

                masks = torch.stack(
                    [mask.to(self.device) for mask in masks_seperate])
                masks_list.append(masks)

        if success_count < num_show:
            raise ValueError(f"检索失败:尝试{top_k}次后仍未找到5个合适的骨架和分割图像")
        
        if out_mask:
            return out_skeletons_list, pose_img_list, match_dict_list, masks_list
        else:
            return out_skeletons_list, pose_img_list, match_dict_list

    def input_skeleton(self, character_list, prompt, skeletons, match_dict, kp_conf):
        # Draw skeleton
        pose_img = np.zeros([self.height, self.width, 3])
        for character in character_list:
            keypoints = np.array(skeletons[match_dict[character.name]])
            try:
                pose_img = draw_bodypose_openpose(pose_img, keypoints, min_conf=kp_conf) # keypoints: (18, 3)
            except Exception as e:
                print(e)

        pose_img = Image.fromarray(pose_img.astype(np.uint8))

        return skeletons, pose_img, match_dict

    def get_mask_from_skeleton(self, skeletons, character_list, match_dict):
        """
        skeletons: [[...],[...], ...] ordered by id
        or 
        skeletons: {id: [...]}

        Skeletons can be indexed by number: 
            skeleton[match_dict[character.name]]
        """
        
        masks = []
        kernel = torch.ones((1, 1, 50, 50), dtype=torch.float32).to(
            'cuda', torch.float32)
        for character in character_list:
            keypoints = np.array(skeletons[match_dict[character.name]])
            mask = draw_mask_fine(keypoints, self.height, self.width, face_ratio=0.6) # (H, W)
            mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
            mask_torch = (F.conv2d(mask_torch, kernel, padding='same') > 0).to(torch.float32)
            masks.append(mask_torch)

        # 如果有2个人，fix the intersection of two masks:mask交集区域用最近邻算法进行分配
        if len(masks) == 2:
            mask0, mask1 = masks[0].squeeze(0), masks[1].squeeze(0) # now is (1, h, w)
            intersection = (mask0 > 0) & (mask1 > 0)

            center0 = torch.nonzero(mask0.squeeze(0)).float().mean(dim=0)
            center1 = torch.nonzero(mask1.squeeze(0)).float().mean(dim=0)
            if center0[0] < center1[0]:  # 注意这里使用索引0
                # mask1更靠下
                mask0[intersection] = 0
                mask1[intersection] = 1
            else:
                # mask0更靠下或两者y坐标相等
                mask0[intersection] = 1
                mask1[intersection] = 0

            masks = [mask0.unsqueeze(0), mask1.unsqueeze(0)] 

        if self.seperate_ref:
            masks_seperate = []
            for character, mask in zip(character_list, masks):
                keypoints = np.array(skeletons[match_dict[character.name]])
                head_face_mask = draw_face_mask(keypoints, self.height, self.width, ratio=0.6).to(self.device)
                head_face_mask = (F.conv2d(head_face_mask.cuda().unsqueeze(0).unsqueeze(0), kernel, padding='same') > 0).squeeze(0).squeeze(0).to(torch.bool)
                mask_head_face = mask * head_face_mask
                mask_body = mask * (~head_face_mask)

                mask = torch.cat([mask_head_face, mask_body], dim=0) # (2, 1, h, w)
                masks_seperate.append(mask)
            masks = masks_seperate

        masks = torch.stack([mask.to(self.device) for mask in masks])
        return masks
    
    def parse_prompt_for_geometry(self, prompt):
        # 1. Parse prompt
        ori_prompts = prompt.lower().split('\n')
        if len(ori_prompts) == 4:
            common_prompt = ori_prompts[0].lower() + ori_prompts[1].lower() #如果有四句话，common prompt是前两句
        else:
            common_prompt = ori_prompts[0].lower()
        
        # 1.1 Convert common prompt for geometry
        common_prompt_for_geometry = ','.join(common_prompt.split(',')[:3])
        return common_prompt_for_geometry
    
    def __call__(
            self,
            character_list: List[Character],
            prompt: List[str],
            bg_prompt: str = None,
            bg_img: Image = None,
            num_samples: int = 4,
            num_inference_steps = 13,
            seed=42, 
            guidance_scale=2.5,
            ref_scale=1.0,
            controlnet_conditioning_scale=1.0,
            pose_img=None,
            masks=None,
            ip_scale=1.0,
            ip_guidance_start=0.0,
            ip_guidance_end=1.0,
        ):
        
        """
        Args:
            prompt: Regional Prompter Format: common prompt \n prompt1 \n prompt2
            e.g. 2characters, Laura kissing Nacho, afternoon, wide shot,\u00a0(2d artistic sketch:1.5), flat color, line art, 
            low saturation, (thin outlines:1.5), trending on Pixiv, (low contrast:1.5), 
            (2d style:1.5), delicate face
            \nLaura, (1Female:1.5), Sales Director, 30s, Caucasian, Auburn Hair, Blazer
            \nNacho, (1Male:1.5), wealthy CEO, European, fit, short dark, suit
       
        """

        set_all_seed(seed)
        start_time_total = time.time()

        # 1. Parse prompt
        ori_prompts = prompt.lower().split('\n')
        if len(ori_prompts) == 4:
            common_prompt = ori_prompts[0].lower() + ori_prompts[1].lower() #如果有四句话，common prompt是前两句
        else:
            common_prompt = ori_prompts[0].lower()
        
        # 1.2 Conver common prompt for character
        # 对于每个character的prompt，把common prompt中的名字都替换成某个character的性别，保证Regional prompter生图的一致性
        character_name = [character.name for character in character_list]
        pattern = re.compile("|".join(re.escape(name) for name in character_name))
        for character_id, character in enumerate(character_list):
            character.converted_prompt = pattern.sub(character.gender, common_prompt)

        # 2. Get Geometry
        masks = masks.repeat(1, 1, num_samples*2, 1, 1) # (num_people, 2(face+body), num_samples*2, h, w)

        # 3. Prepare Ref Img and Prompt
        ref_img_list = []
        prompts = []
        for character in character_list:
            ref_img_list.append(character.ref_img)
            prompts.append(character.converted_prompt)

        # When multi_ref 如果每个character人脸有多张ref img，则需要记录具体ref img数量
        if self.multi_ref_method is not None:
            if self.multi_ref_method == 'stack':
                if self.seperate_ref: # Remove body ref 如果人脸和衣服分开ref，这里只记录人脸的数量，排除衣服图片，所以减一
                    multi_ref_num = [len(ref_img) - 1 for ref_img in ref_img_list]
                else:
                    multi_ref_num = [len(ref_img) for ref_img in ref_img_list]
    
        # Background 不管是否有背景ref img和prompt我们都将其加进来（空字符串或黑图）以便于后续的batch运算
        if bg_prompt is not None:
            prompts.append(bg_prompt)
        else:
            prompts.append('')

        if bg_img is not None:
            if self.multi_ref_method == 'stack':
                ref_img_list.append([bg_img])
            else:
                ref_img_list.append(bg_img) # TODO: Adapt bg img when multi ref
        else:
            if self.multi_ref_method == 'stack':
                ref_img_list.append([Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))])
            else:
                ref_img_list.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        multi_ref_num.append(1) # Add bg ref， 背景也是作为ref img所以需要记录进ref img数量

        images_list = self.ip_model.generate(
            pil_image=ref_img_list, 
            prompt=prompts, 
            ori_prompt=common_prompt,
            num_samples=num_samples, 
            num_inference_steps=num_inference_steps, 
            seed=seed, 
            width=self.width, 
            height=self.height,
            image=pose_img, 
            controlnet_conditioning_scale=controlnet_conditioning_scale, 
            guidance_scale=guidance_scale,
            scale=ref_scale,
            cross_attention_kwargs={
                'masks':masks, 
                "use_bg_prompt": bg_prompt is not None,
                "use_bg_img": bg_img is not None,       
                "multi_ref_num": multi_ref_num if self.multi_ref_method is not None and self.multi_ref_method == 'stack' else None      
                },
            ip_scale=ip_scale,
            ip_guidance_start=ip_guidance_start,
            ip_guidance_end=ip_guidance_end,
            )
        end_time = time.time()
        total_time = end_time - start_time_total
        print('total_time:', total_time)
        return images_list

