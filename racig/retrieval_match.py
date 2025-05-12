import torch
import sys, os
import warnings
sys.path.append(os.path.dirname(sys.path[0]))

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
import cv2
import time
from openai import AzureOpenAI
import json
import pickle
from .utils import pil_images_to_matplotlib_figure
import base64
import io
from grounding_dino.groundingdino import build_groundingdino_classification
from grounding_dino.utils import clean_state_dict
import torchvision.transforms.functional as F


class Retriever:
    def __init__(self, clip_model_name, index_path, json_dict_path, data_root, device):
        self.model = CLIPModel.from_pretrained(clip_model_name).to(device, dtype=torch.float16)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.index_path = index_path
        
        self.json_root = os.path.join(data_root, 'json')
        self.img_root = os.path.join(data_root, 'img')
        self.skeleton_root = os.path.join(data_root, 'skeleton')
        self.seg_root = os.path.join(data_root, 'seg')
        self.head_seg_root = os.path.join(data_root, 'head_seg_npz')
        self.face_seg_root = os.path.join(data_root, 'face_seg_npz')

        index_path_1 = self.index_path + '_1character.npy'
        index_path_2 = self.index_path + '_2character.npy'
        self.image_features = [torch.from_numpy(np.load(index_path_1)).to('cuda', dtype=torch.float16), torch.from_numpy(np.load(index_path_2)).to('cuda', dtype=torch.float16)]
        self.image_features_for_excel = self.image_features.copy()

        json_dict_path_1 = json_dict_path + '_1character.pkl'
        json_dict_path_2 = json_dict_path + '_2character.pkl'
        self.json_paths = []
        with open(json_dict_path_1, 'rb') as f:
            json_paths_dict_1 = pickle.load(f)
        json_paths_1 = []
        for key,value in json_paths_dict_1.items():
            value = value.replace("json_w_gender_age", "json")
            json_paths_1.append(value)
        self.json_paths.append(json_paths_1)

        with open(json_dict_path_2, 'rb') as f:
            json_paths_dict_2 = pickle.load(f)
        json_paths_2 = []
        for key, value in json_paths_dict_2.items():
            value = value.replace("json_w_gender_age", "json")
            json_paths_2.append(value)
        self.json_paths.append(json_paths_2)
            
        # self.json_paths = glob.glob(os.path.join(self.json_root, '**/*.json'), recursive=True)
        self.img_paths = [[p.replace(self.json_root, self.img_root)[:-5]+'.jpg' if 'Reelshot' in p else p.replace('/json', '/img')[:-5]+'.jpg' for p in self.json_paths[0] ], [p.replace(self.json_root, self.img_root)[:-5]+'.jpg' if 'Reelshot' in p else p.replace('/json', '/img')[:-5]+'.jpg' for p in self.json_paths[1] ]]
        
        self.skeleton_paths = [[p.replace(self.json_root, self.skeleton_root)[:-5]+'.json' if 'Reelshot' in p else p.replace('/json', '/skeleton')[:-5]+'.json' for p in self.json_paths[0]], [p.replace(self.json_root, self.skeleton_root)[:-5]+'.json' if 'Reelshot' in p else p.replace('/json', '/skeleton')[:-5]+'.json' for p in self.json_paths[1]]]
        
        self.seg_paths = [[p.replace(self.json_root, self.seg_root)[:-5]+'.npy' if 'Reelshot' in p else p.replace('/json', '/seg')[:-5]+'.npy' for p in self.json_paths[0]], [p.replace(self.json_root, self.seg_root)[:-5]+'.npy' if 'Reelshot' in p else p.replace('/json', '/seg')[:-5]+'.npy' for p in self.json_paths[1]]]
        
        self.face_seg_paths = [[p.replace(self.json_root, self.face_seg_root)[:-5]+'.npz' if 'Reel_shot' in p else p.replace('/json', '/face_seg_npz')[:-5]+'.npz' for p in self.json_paths[0]], [p.replace(self.json_root, self.face_seg_root)[:-5]+'.npz' if 'Reel_shot' in p else p.replace('/json', '/face_seg_npz')[:-5]+'.npz' for p in self.json_paths[1]]]
        
        self.head_seg_paths = [[p.replace(self.json_root, self.head_seg_root)[:-5]+'.npz' if 'Reel_shot' in p else p.replace('/json', '/head_seg_npz')[:-5]+'.npz' for p in self.json_paths[0]], [p.replace(self.json_root, self.face_seg_root)[:-5]+'.npz' if 'Reel_shot' in p else p.replace('/json', '/head_seg_npz')[:-5]+'.npz' for p in self.json_paths[1]]]

    def search(self, text, k):
        st = time.time()
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        text_features = self.model.get_text_features(**inputs).to(self.model.device)
        et = time.time()
        print('retrieve embedding time:', et-st)

        try: 
            num_character = int(text.strip().split(',')[0][0])
        except: 
            num_character = 1
        image_features = self.image_features[num_character-1]

        # 进行向量相似度搜索
        st = time.time()
        similarities = torch.matmul(image_features, text_features.T)     
        _, indices = torch.topk(similarities, k, dim=0)
        et = time.time()

        # 根据索引返回top k的图片路径
        searched_img_paths = [self.img_paths[num_character-1][i] for i in indices[:,0]]
        searched_json_paths = [self.json_paths[num_character-1][i] for i in indices[:,0]]
        searched_skeleton_paths = [self.skeleton_paths[num_character-1][i] for i in indices[:,0]]
        searched_seg_paths = [self.seg_paths[num_character-1][i] for i in indices[:,0]]
        searched_face_seg_paths = [self.face_seg_paths[num_character-1][i] for i in indices[:,0]]
        searched_head_seg_paths = [self.head_seg_paths[num_character-1][i] for i in indices[:,0]]

        return {
            "img_paths": searched_img_paths,
            "json_paths": searched_json_paths,
            "skeleton_paths": searched_skeleton_paths,
            "seg_paths": searched_seg_paths,
            "face_seg_paths": searched_face_seg_paths,
            "head_seg_paths": searched_head_seg_paths,
        }

    def search_for_excel(self, text, k, rank, num_person=1):
         # 对输入文本进行embedding
        st = time.time()
        inputs = self.processor(text=text, return_tensors="pt").to(f"cuda:{rank}")
        text_features = self.model.get_text_features(**inputs).to(f"cuda:{rank}")
        et = time.time()
        print('retrieve embedding time:', et-st)

        num_character = num_person
        image_features = self.image_features_for_excel[num_character-1]

        # 进行向量相似度搜索
        st = time.time()
        similarities = torch.matmul(image_features, text_features.T)     
        _, indices = torch.topk(similarities, k, dim=0)
        mask = torch.ones(image_features.shape[0], dtype=bool).to(indices.device)
        mask[indices[:,0]] = False
        self.image_features_for_excel[num_character-1] = image_features[mask]

        et = time.time()
        # print('retrieve search time:', et-st)

        # 根据索引返回top k的图片路径
        searched_img_paths = [self.img_paths[num_character-1][i] for i in indices[:,0]]
        searched_json_paths = [self.json_paths[num_character-1][i] for i in indices[:,0]]
        searched_skeleton_paths = [self.skeleton_paths[num_character-1][i] for i in indices[:,0]]
        searched_seg_paths = [self.seg_paths[num_character-1][i] for i in indices[:,0]]
        searched_face_seg_paths = [self.face_seg_paths[num_character-1][i] for i in indices[:,0]]
        searched_head_seg_paths = [self.head_seg_paths[num_character-1][i] for i in indices[:,0]]

        return {
            "img_paths": searched_img_paths,
            "json_paths": searched_json_paths,
            "skeleton_paths": searched_skeleton_paths,
            "seg_paths": searched_seg_paths,
            "face_seg_paths": searched_face_seg_paths,
            "head_seg_paths": searched_head_seg_paths,
        }

    def retrieve_and_display_images(self, text, k=10):
        start_time = time.time()
        searched_paths = self.search_images(text, k)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Search Time: {total_time} seconds")
        
        pil_images = [Image.open(path) for path in searched_paths]
        fig = pil_images_to_matplotlib_figure(pil_images)
        fig.savefig("preview_img.png", bbox_inches='tight')


class Matcher_Grounding_DINO:
    def __init__(
            self, 
            config_path="grounding_dino/config/cfg_cls.py", 
            ckpt_path="models/action_direction_dino/checkpoint_best_regular.pth",
            device='cuda'
            ):

        self.device = device
        try:
            # from groundingdino import _C
            import MultiScaleDeformableAttention as _C
        except:
            self.device = 'cpu'
            warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")

        self.model = build_groundingdino_classification(config_path).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(clean_state_dict(checkpoint['model']))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225] 

    def match_subjects(self, sentence, img:Image, segmap, segments, character_list):
        sentence = sentence.lower()
        img_np = np.array(img)
        w, h = img_np.shape[1], img_np.shape[0]

        diagonal = np.sqrt(w**2 + h**2)

        base_fontScale = 0.005 * diagonal
        base_thickness = int(0.01 * diagonal)
        
        base_fontScale = max(1, min(base_fontScale, 15))
        base_thickness = max(1, min(base_thickness, 50))
        fontScale = base_fontScale 
        thickness = int(base_thickness)
        id_color_map = {0:[255,0,0], 1:[0,255,0], 2:[0,0,255], 3:[0,255,255] , 4:[255,255,0] ,5:[255,0,255], 6:[125, 125, 125], 7:[123, 0, 255], 8:[234, 0, 123]}
        for segment, character in zip(segments, character_list):
            seg_id = segment['id']

            sentence = sentence.replace(character.name, f'subject{seg_id}')

            segmap_id = (segmap == seg_id)
            coords = np.argwhere(segmap_id)
            if len(coords) > 0:
                centroid = coords.mean(axis=0)
            x_mid, y_mid = int(centroid[1]), int(centroid[0])
            cv2.putText(img_np, str(seg_id), (x_mid-30, y_mid+30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, color=id_color_map[seg_id], thickness=thickness)
        input_img = F.normalize(torch.from_numpy(img_np).permute(2,0,1)/255, mean=self.mean, std=self.std).to(self.device)
        sentence = sentence + '.' if sentence[-1] != '.' else sentence
        output =  self.model([input_img], captions=[sentence])  
        if torch.max(output, 1)[1].item() == 1:
            match_dict = {character_list[0].name: segments[0]['id'], character_list[1].name: segments[1]['id']}
        else:
            match_dict = {character_list[0].name: segments[1]['id'], character_list[1].name: segments[0]['id']}
        
        debug = False
        if debug:
            return match_dict, img_np, sentence, output
        else:
            return match_dict