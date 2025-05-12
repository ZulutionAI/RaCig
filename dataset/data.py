import os
import torch
# from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop
import glob
import json
import numpy as np
import random
from copy import deepcopy
from pathlib import Path
from PIL import Image
import re
from insightface.app import FaceAnalysis
import cv2
import scipy
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms_face,
        object_transforms_body,
        object_processor,
        device=None,
        max_num_objects=4,
        num_image_tokens=1,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_types=None,
        split="all",
        min_num_objects=None,
        balance_num_objects=False,
        args=None
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms_body = object_transforms_body
        self.object_transforms_face = object_transforms_face
        self.object_processor = object_processor
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.num_image_tokens = num_image_tokens
        self.object_appear_prob = object_appear_prob
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.object_types = object_types
        self.multi = args.multi
        self.add_id_embeds = args.add_id_embeds
        self.use_controlnet = args.use_controlnet
        self.control_cond = args.control_cond

        self.json_root = os.path.join(root, 'json')
        self.img_root = os.path.join(root, 'img')
        self.seg_root = os.path.join(root, 'seg')
        self.face_seg_root = os.path.join(root, 'face_seg_npz')
        self.head_seg_root = os.path.join(root, 'head_seg_npz')
        
        self.json_paths = glob.glob(os.path.join(self.json_root, '**/*.json'), recursive=True)
        
        self.img_paths = [p.replace(self.json_root, self.img_root)[:-5]+'.jpg' for p in self.json_paths]
        self.seg_paths = [os.path.dirname(p.replace(self.json_root, self.seg_root))+'/'+os.path.basename(p)[:-5]+'.npy' for p in self.json_paths]
        self.face_seg_paths = [os.path.dirname(p.replace(self.json_root, self.face_seg_root))+'/'+os.path.basename(p)[:-5]+'.npz' for p in self.json_paths]
        self.head_seg_paths = [os.path.dirname(p.replace(self.json_root, self.head_seg_root))+'/'+os.path.basename(p)[:-5]+'.npz' for p in self.json_paths]

        self.face_model = FaceAnalysis(name='antelopev2', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_model.prepare(ctx_id=0,)
        if args.cut_dataset:
            # Generate a list of indices and shuffle it
            indices = list(range(len(self.json_paths)))
            random.shuffle(indices)

            self.json_paths = [self.json_paths[i] for i in indices][:10]
            self.img_paths = [self.img_paths[i] for i in indices][:10]
            self.seg_paths = [self.seg_paths[i] for i in indices][:10]
        
        if self.use_controlnet and self.control_cond == 'skeleton':
            self.skeleton_root = os.path.join(root, 'skeleton')
            self.skeleton_paths = [p.replace(self.json_root, self.skeleton_root)[:-5]+'.json' for p in self.json_paths]

        if self.add_id_embeds:
            self.face_embedding_root = os.path.join(root, 'face_embedding')
            self.face_embedding_paths = [p.replace(self.json_root, self.face_embedding_root)[:-5]+'.npy' for p in self.json_paths]


        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)


        if min_num_objects is not None:
            print(f"Filtering images with less than {min_num_objects} objects")
            filtered_image_ids = []
            for image_id in tqdm(self.image_ids):
                chunk = image_id[:5]
                info_path = os.path.join(self.root, chunk, image_id + ".json")
                with open(info_path, "r") as f:
                    info_dict = json.load(f)
                segments = info_dict["segments"]

                if self.object_types is not None:
                    segments = [
                        segment
                        for segment in segments
                        if segment["coco_label"] in self.object_types
                    ]

                if len(segments) >= min_num_objects:
                    filtered_image_ids.append(image_id)
            self.image_ids = filtered_image_ids


    def __len__(self):
        return len(self.json_paths)

    @torch.no_grad()
    def preprocess(self, image, info_dict, segmap, face_segmap, head_segmap, skeleton, image_id):

        caption_detail = info_dict["caption_detail"]
        caption_bg = info_dict["caption_bg"]
        caption = caption_detail + ', Background: ' + caption_bg
        caption = caption.lower()
        segments = info_dict["segments"]

        segid_subjectid_dict = {}
        for subject_id, segment in enumerate(segments):
            segid_subjectid_dict[segment["id"]] = subject_id
            
        if self.object_types is not None:
            segments = [
                segment
                for segment in segments
                if segment["coco_label"] in self.object_types
            ]

        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            segments = []
        elif prob < self.uncondition_prob + self.text_only_prob:
            segments = []
        else:
            segments = [
                segment
                for segment in segments
                if random.random() < self.object_appear_prob
            ]

        try:
            segments = [ segment for segment in segments if segment['is_clear'] == 'yes']
        except:
            segments = segments
            
        if len(segments) > self.max_num_objects:
            # random sample objects
            segments = random.sample(segments, self.max_num_objects)
        if self.multi:
            segments = sorted(segments, key=lambda x: x["id"])
        else:
            segments = sorted(segments, key=lambda x: x["caption_segs"][0][-1])

        background = self.object_processor.get_background(image)

        # draw bbox image!
        if self.use_controlnet:
            if self.control_cond == 'bbox':
                def draw_bbox(image, bbox, color=(0, 255, 0), thickness=20):
                    h1, w1, h2, w2 = bbox
                    out = cv2.rectangle(image, (h1, w1), (h2, w2), color, thickness)
                    return out
                
                _, h, w = image.shape
                
                draw_info = []
                for s_id, segment in enumerate(segments):
                    # print(s_id)
                    color = cv2.applyColorMap(np.array([[255*(s_id+1)/self.max_num_objects]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0]
                    bbox = segment['bbox']
                    # print(f"{s_id}, {bbox}")
                    color=tuple(map(int, color))
                    area = abs((bbox[0]-bbox[2])*(bbox[1]-bbox[3]))
                    draw_info.append((s_id, color, area, bbox))
                
                draw_info.sort(key=lambda x: x[2], reverse=True)
                bbox_img = np.zeros((h, w, 3), dtype=np.uint8)

                for _,color,_,bbox in draw_info:
                    bbox_img = draw_bbox(bbox_img, bbox, color=color, thickness=-1)
        

                bbox_img = torch.from_numpy(np.array(bbox_img)).permute(2,0,1).to(image.device)
                pixel_values, transformed_segmap, bbox_img = self.train_transforms(image, segmap, bbox_img)

            elif self.control_cond == 'skeleton':
                # draw skeleton !
                def draw_bodypose(canvas: np.ndarray, keypoints: np.ndarray, min_conf: float) -> np.ndarray:
                    H, W, C = canvas.shape
                    # automatically adjust the thickness of the skeletons
                    if max(W, H) < 500:
                        ratio = 1.0
                    elif max(W, H) >= 500 and max(W, H) < 1000:
                        ratio = 2.0
                    elif max(W, H) >= 1000 and max(W, H) < 2000:
                        ratio = 3.0
                    elif max(W, H) >= 2000 and max(W, H) < 3000:
                        ratio = 4.0
                    elif max(W, H) >= 3000 and max(W, H) < 4000:
                        ratio = 5.0
                    elif max(W, H) >= 4000 and max(W, H) < 5000:
                        ratio = 6.0
                    else:
                        ratio = 7.0
                    stickwidth = 4
                    # connections and colors
                    limbSeq = [
                        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                        [1, 16], [16, 18]]

                    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],[85, 255, 0], 
                            [0, 255, 0],[0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],[0, 85, 255], 
                            [0, 0, 255], [85, 0, 255], [170, 0,255], [255, 0, 255],[255, 0, 170], [255, 0, 85]]

                    # draw the links
                    for (k1, k2), color in zip(limbSeq, colors):
                        cur_canvas = canvas.copy()
                        keypoint1 = keypoints[k1-1, :]
                        keypoint2 = keypoints[k2-1, :]

                        if keypoint1[-1] < min_conf or keypoint2[-1] < min_conf:
                            continue

                        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
                        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        import math
                        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
                        cv2.fillConvexPoly(cur_canvas, polygon, [int(float(c)) for c in color])
                        canvas = cv2.addWeighted(canvas, 0.3, cur_canvas, 0.7, 0)
                    # draw the points
                    for id, color in zip(range(keypoints.shape[0]), colors):
                        keypoint = keypoints[id, :]
                        if keypoint[-1]<min_conf:
                            continue

                        x, y = keypoint[0], keypoint[1]
                        x = int(x * W)
                        y = int(y * H)
                        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

                    return canvas

                _, H, W = image.shape
                
                # black background of the skeleton
                canvas = np.zeros(image.shape).transpose(1,2,0)

                if len(skeleton) > 0:
                    items = [ele['skeleton'] for ele in skeleton if ele['skeleton'] is not None]
                    for person in items:
                        keypoints = np.zeros((18,3))
                        idx = 0
                        for point, score in zip(person['keypoints'], person['keypoint_scores']):
                                normalized_x = point[0] / W
                                normalized_y = point[1] / H
                                keypoints[idx] = [normalized_x, normalized_y, score]
                                idx += 1
                        neck = (keypoints[5,:] + keypoints[6,:])/2
                        keypoints[idx] = neck
                        # transform mmpose keypoints to openpose keypoints
                        openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
                        mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
                        new_keypoints = keypoints[:, :]
                        new_keypoints[openpose_idx, :] = keypoints[mmpose_idx, :]
                        # draw skeleton of current person
                        try:
                            canvas = draw_bodypose(canvas, new_keypoints, 0.1)
                        except:
                            print("Draw skeleton failed, use black canvas")
                            canvas = canvas
                        
                skeleton_ori = torch.from_numpy(np.array(canvas)).permute(2,0,1).to(image.device)

                pixel_values, transformed_segmap, skeleton = self.train_transforms(image, segmap, skeleton_ori)
                _, transformed_face_segmap, _ = self.train_transforms(image, face_segmap, skeleton_ori)
                _, transformed_head_segmap, _ = self.train_transforms(image, head_segmap, skeleton_ori)
        else:
            pixel_values, transformed_segmap = self.train_transforms(image, segmap)
        # face_embeddings = []
        object_face_segmaps = []
        object_head_segmaps = []
        object_pixel_values_body = []
        object_pixel_values_face = []
        for segment in segments:
            id = segment["id"]
            bbox = np.array(segment["bbox"])  # [h1, w1, h2, w2]
            bbox[bbox<0] = 0 # Fix negative bug 
            object_image = self.object_processor(
                deepcopy(image), background, segmap, id, bbox
            )
            object_pixel_values_body.append(self.object_transforms_body(object_image))
            object_pixel_values_face.append(self.object_transforms_face(object_image))
            object_segmaps.append(transformed_segmap == id)
            object_face_segmaps.append(transformed_face_segmap == id)
            object_head_segmaps.append(transformed_head_segmap == id)

        input_ids = self.tokenizer.encode(caption, max_length=77, truncation=True)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (
            77 - len(input_ids)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        num_objects = len(segments)
        object_pixel_values_face = object_pixel_values_face[:num_objects]
        object_pixel_values_body = object_pixel_values_body[:num_objects]
        object_head_segmaps = object_head_segmaps[:num_objects]
        object_face_segmaps = object_face_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values_face[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values_face += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_pixel_values_body += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_head_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_face_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]

        object_pixel_values_body = torch.stack(
            object_pixel_values_body
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values_body = object_pixel_values_body.to(
            memory_format=torch.contiguous_format
        ).float()


        object_pixel_values_face = torch.stack(
            object_pixel_values_face
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values_face = object_pixel_values_face.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            object_segmaps
        ).float()  # [max_num_objects, 256, 256]

        object_head_segmaps = torch.stack(
            object_head_segmaps
        ).float()  # [max_num_objects, 256, 256]
        
        object_face_segmaps = torch.stack(
            object_face_segmaps
        ).float()  # [max_num_objects, 256, 256]
        
        return {
            "bbox_img": bbox_img if self.use_controlnet and self.control_cond=='bbox' else None,
            "skeleton": skeleton if self.use_controlnet and self.control_cond=='skeleton' else None,
            "segid_subjectid_dict": segid_subjectid_dict,
            "caption": caption,
            "image":image,
            "segmap":segmap,
            "pixel_values": pixel_values, # (3, h, w)
            "transformed_segmap": transformed_segmap, # (h ,w)
            "input_ids": input_ids, # (1, seq_len)
            "object_pixel_values_body": object_pixel_values_body, # (max_num_objects, 3, 224, 224)
            "object_pixel_values_face": object_pixel_values_face, # (max_num_objects, 3, 224, 224)
            "object_segmaps": object_segmaps, # (max_num_objects, 224, 224)
            "object_face_segmaps": object_face_segmaps, # (max_num_objects, 224, 224)
            "object_head_segmaps": object_head_segmaps, # (max_num_objects, 224, 224)
            "num_objects": torch.tensor(num_objects),
            "image_ids": torch.tensor(image_id),
        }

    def __getitem__(self, idx):


        image_path = self.img_paths[idx]
        segmap_path = self.seg_paths[idx]
        face_segmap_path = self.face_seg_paths[idx]
        head_segmap_path = self.head_seg_paths[idx]
        info_path = self.json_paths[idx]
        face_embeddings = None
        if self.add_id_embeds:
            face_embedding_path = self.face_embedding_paths[idx]
            face_embeddings = torch.from_numpy(np.load(face_embedding_path))

        skeleton = None
        if self.use_controlnet and self.control_cond == 'skeleton':
            with open(self.skeleton_paths[idx]) as json_file:
                skeleton = json.load(json_file)

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2,0,1)

        with open(info_path, "r") as f:
            info_dict = json.load(f)
        segmap = torch.from_numpy(np.load(segmap_path))
        try:
            face_segmap = torch.from_numpy(scipy.sparse.load_npz(face_segmap_path).toarray())
            head_segmap = torch.from_numpy(scipy.sparse.load_npz(head_segmap_path).toarray())
        except:
            face_segmap = torch.from_numpy(np.load(segmap_path))
            head_segmap = torch.from_numpy(np.load(segmap_path))

        if self.device is not None:
            image = image.to(self.device)
            segmap = segmap.to(self.device)
            face_segmap = face_segmap.to(self.device)
            head_segmap = head_segmap.to(self.device)
        return_dict = self.preprocess(image, info_dict, segmap, face_segmap, head_segmap, skeleton, idx)
        return_dict['face_embeddings'] = face_embeddings
        return return_dict

def collate_fn(examples):
    face_embeddings = None
    if examples[0]["face_embeddings"] is not None :
        face_embeddings = torch.stack([example["face_embeddings"] for example in examples])

    bbox_img = None
    if examples[0]["bbox_img"] is not None:
        bbox_img = torch.stack([example["bbox_img"] for example in examples])

    skeleton = None
    if examples[0]["skeleton"] is not None:
        skeleton = torch.stack([example["skeleton"] for example in examples])

    segid_subjectid_dict = [example["segid_subjectid_dict"] for example in examples]
    caption = [example["caption"] for example in examples]
    image = [example["image"] for example in examples]

    segmap = [example["segmap"] for example in examples]

    transformed_segmap = torch.stack(
        [example["transformed_segmap"] for example in examples]
    )

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    image_ids = torch.stack([example["image_ids"] for example in examples])


    object_pixel_values_body = torch.stack(
        [example["object_pixel_values_body"] for example in examples]
    )
    object_pixel_values_face = torch.stack(
        [example["object_pixel_values_face"] for example in examples]
    )
    object_segmaps = torch.stack([example["object_segmaps"] for example in examples])
    object_head_segmaps = torch.stack([example["object_head_segmaps"] for example in examples])
    object_face_segmaps = torch.stack([example["object_face_segmaps"] for example in examples])

    num_objects = torch.stack([example["num_objects"] for example in examples])

    
    outputs =  {
        "skeleton": skeleton if skeleton is not None else None,
        "bbox_img": bbox_img if bbox_img is not None else None,
        "face_embeddings": face_embeddings if face_embeddings is not None else None,
        "segid_subjectid_dict":segid_subjectid_dict,
        "caption": caption,
        "image":image,
        "segmap":segmap,
        "pixel_values": pixel_values,
        "transformed_segmap": transformed_segmap,
        "input_ids": input_ids,
        "object_pixel_values_body": object_pixel_values_body,
        "object_pixel_values_face": object_pixel_values_face,
        "object_segmaps": object_segmaps,
        "object_head_segmaps":object_head_segmaps,
        "object_face_segmaps":object_face_segmaps,
        "num_objects": num_objects,
        "image_ids": image_ids,
    }
    return outputs

def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )

    return dataloader