import os
from typing import List
import time
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn.functional as F
from .utils import is_torch2_available, get_generator
import numpy as np

import random

if is_torch2_available():
    from .attention_processor_mask import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor_mask import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor_mask import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
    from .attention_processor_mask import IPAttnProcessor2_0_RP, IPAttnProcessor2_0_RP_Double_Adapter
else:
    from .attention_processor_mask import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlusXL_RP(IPAdapter):
    """SDXL"""
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, height=1344, width=768, expand_mask=False, masked_attn=False):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.height = height
        self.width = width  
        self.expand_mask = expand_mask
        self.masked_attn = masked_attn

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor2_0_RP(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    height=self.height,
                    width=self.width,
                    masked_attn=self.masked_attn
                ).to(self.device, dtype=torch.float16)
                
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj_model": {}, "unet": {}, "controlnet": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("unet.") and 'attn2' in key and 'processor' in key :
                        state_dict["unet"][key.replace("unet.", "")] = f.get_tensor(key)
                    elif key.startswith("controlnet."):
                        state_dict["controlnet"][key.replace("controlnet.", "")] = f.get_tensor(key)

            self.image_proj_model.load_state_dict(state_dict["image_proj_model"])
            self.pipe.unet.load_state_dict(state_dict["unet"], strict=False)
                    
            self.pipe.controlnet.load_state_dict(state_dict["controlnet"])

        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])


    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_all_seed(self, seed=42):
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

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        self.set_all_seed()
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def get_embeddings(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        text_embeds_all = (None, None, None, None)
        text_embeds_seperate = (None, None, None, None)
        if prompt is not None:
            text_embeds_seperate = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            
            text_embeds_all = self.pipe.encode_prompt(
                prompt[0],
                negative_prompt=negative_prompt[0],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            seq_len = text_embeds_seperate[0].shape[1]
            prompt_embeds = torch.cat((
                text_embeds_seperate[0].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1), 
                image_prompt_embeds), dim=1)
            chunks = prompt_embeds.chunk(prompt_embeds.shape[0])
            # prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples]), dim=1) for i in range(num_samples)])
            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            prompt_embeds = torch.cat(chunk_prompt)
                
            #prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples], chunks[i + 2*num_samples]), dim=1) for i in range(num_samples)])

            negative_prompt_embeds = torch.cat((
                text_embeds_seperate[1].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1),
                uncond_image_prompt_embeds), dim=1)
            chunks = negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0])
            # negative_prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples]), dim=1) for i in range(num_samples)])
            #negative_prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples], chunks[i + 2*num_samples]), dim=1) for i in range(num_samples)])
            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            negative_prompt_embeds = torch.cat(chunk_prompt)

        pooled_prompt_embeds = text_embeds_all[2].repeat(num_samples, 1)
        pooled_negative_prompt_embeds = text_embeds_all[3].repeat(num_samples, 1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(prompt)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        text_embeds_all = (None, None, None, None)
        text_embeds_seperate = (None, None, None, None)
        if prompt is not None:
            self.set_all_seed(seed)

            text_embeds_seperate = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            
            text_embeds_all = self.pipe.encode_prompt(
                prompt[0] + prompt[-1],
                negative_prompt=negative_prompt[0],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            seq_len = text_embeds_seperate[0].shape[1]
            prompt_embeds = torch.cat((
                text_embeds_seperate[0].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1), 
                image_prompt_embeds), dim=1)
            chunks = prompt_embeds.chunk(prompt_embeds.shape[0])

            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            prompt_embeds = torch.cat(chunk_prompt)

            negative_prompt_embeds = torch.cat((
                text_embeds_seperate[1].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1),
                uncond_image_prompt_embeds), dim=1)
            chunks = negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0])

            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            negative_prompt_embeds = torch.cat(chunk_prompt)

        pooled_prompt_embeds = text_embeds_all[2].repeat(num_samples, 1)
        pooled_negative_prompt_embeds = text_embeds_all[3].repeat(num_samples, 1)
        
        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            seed=seed,
            **kwargs,
        ).images

        return images

class IPAdapterPlusXL_RP_MultiRef(IPAdapter):
    """SDXL"""
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, height=1344, width=768, ref_method='average'):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.height = height
        self.width = width

        assert ref_method in ['average', 'stack']
        self.ref_method = ref_method
        
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor2_0_RP(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    height=self.height,
                    width=self.width,
                ).to(self.device, dtype=torch.float16)
                
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj_model": {}, "unet": {}, "controlnet": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("unet.") and 'attn2' in key and 'processor' in key :
                        state_dict["unet"][key.replace("unet.", "")] = f.get_tensor(key)
                    elif key.startswith("controlnet."):
                        state_dict["controlnet"][key.replace("controlnet.", "")] = f.get_tensor(key)

            self.image_proj_model.load_state_dict(state_dict["image_proj_model"])
                    
            self.pipe.controlnet.load_state_dict(state_dict["controlnet"])

        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_all_seed(self, seed=42):
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

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_image_embeds_average(self, pil_image):
        """
        pil_image: [[img1, img2], [img1, img2, img3]]
        """
        assert isinstance(pil_image, list) and isinstance(pil_image[0], list)
        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image_character in pil_image:
            clip_image = self.clip_image_processor(images=pil_image_character, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
            image_prompt_embeds_list.append(image_prompt_embeds.mean(dim=0 ,keepdim=True))
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds.mean(dim=0 ,keepdim=True))

        return torch.cat(image_prompt_embeds_list, dim=0), torch.cat(uncond_image_prompt_embeds_list, dim=0)

    def get_image_embeds_stack(self, pil_image):
        """
        pil_image: [[img1, img2], [img1, img2, img3]]
        """
        assert isinstance(pil_image, list) and isinstance(pil_image[0], list)
        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        max_imgs_num = max([len(pil_image_character) for pil_image_character in pil_image])
        for pil_image_character in pil_image:
            if len(pil_image_character) < max_imgs_num:
                for _ in range(max_imgs_num - len(pil_image_character)):
                    pil_image_character.append(Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)))
            clip_image = self.clip_image_processor(images=pil_image_character, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
            image_prompt_embeds_list.append(image_prompt_embeds.view(1, self.num_tokens*max_imgs_num, -1))
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds.view(1, self.num_tokens*max_imgs_num, -1))

        return torch.cat(image_prompt_embeds_list, dim=0), torch.cat(uncond_image_prompt_embeds_list, dim=0)

    def get_embeddings(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        if self.ref_method == 'average':
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_average(pil_image)
        elif self.ref_method == 'stack':
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        text_embeds_all = (None, None, None, None)
        text_embeds_seperate = (None, None, None, None)
        if prompt is not None:
            text_embeds_seperate = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            
            text_embeds_all = self.pipe.encode_prompt(
                prompt[0],
                negative_prompt=negative_prompt[0],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            seq_len = text_embeds_seperate[0].shape[1]
            prompt_embeds = torch.cat((
                text_embeds_seperate[0].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1), 
                image_prompt_embeds), dim=1)
            chunks = prompt_embeds.chunk(prompt_embeds.shape[0])
            # prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples]), dim=1) for i in range(num_samples)])
            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            prompt_embeds = torch.cat(chunk_prompt)
                
            #prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples], chunks[i + 2*num_samples]), dim=1) for i in range(num_samples)])

            negative_prompt_embeds = torch.cat((
                text_embeds_seperate[1].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1),
                uncond_image_prompt_embeds), dim=1)
            chunks = negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0])
            # negative_prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples]), dim=1) for i in range(num_samples)])
            #negative_prompt_embeds = torch.cat([torch.cat((chunks[i], chunks[i + num_samples], chunks[i + 2*num_samples]), dim=1) for i in range(num_samples)])
            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            negative_prompt_embeds = torch.cat(chunk_prompt)

        pooled_prompt_embeds = text_embeds_all[2].repeat(num_samples, 1)
        pooled_negative_prompt_embeds = text_embeds_all[3].repeat(num_samples, 1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(prompt)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        if self.ref_method == 'average':
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_average(pil_image)
        elif self.ref_method == 'stack':
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_stack(pil_image)
        
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        text_embeds_all = (None, None, None, None)
        text_embeds_seperate = (None, None, None, None)
        if prompt is not None:
            self.set_all_seed(seed)

            text_embeds_seperate = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            
            text_embeds_all = self.pipe.encode_prompt(
                prompt[0] + prompt[-1],
                negative_prompt=negative_prompt[0],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            seq_len = text_embeds_seperate[0].shape[1]
            prompt_embeds = torch.cat((
                text_embeds_seperate[0].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1), 
                image_prompt_embeds), dim=1)
            chunks = prompt_embeds.chunk(prompt_embeds.shape[0])

            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            prompt_embeds = torch.cat(chunk_prompt)

            negative_prompt_embeds = torch.cat((
                text_embeds_seperate[1].repeat(1, num_samples, 1).view(bs_embed * num_samples, seq_len, -1),
                uncond_image_prompt_embeds), dim=1)
            chunks = negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0])

            chunk_prompt = []
            for i in range(num_samples):
                chunk_prompt_one = []
                for prompt_idx in range(len(prompt)):
                    chunk_prompt_one.append(chunks[i+prompt_idx*num_samples])
                chunk_prompt.append(torch.cat(chunk_prompt_one, dim=1))
            negative_prompt_embeds = torch.cat(chunk_prompt)

        pooled_prompt_embeds = text_embeds_all[2].repeat(num_samples, 1)
        pooled_negative_prompt_embeds = text_embeds_all[3].repeat(num_samples, 1)
        
        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
    
class IPAdapterPlusXL_RP_MultiRef_Double_Adapter(IPAdapter):
    """SDXL"""
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, ip_ckpt_body, device, num_tokens=4, height=1344, width=768, ref_method='average'):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.ip_ckpt_body = ip_ckpt_body

        self.num_tokens = num_tokens

        self.height = height
        self.width = width

        assert ref_method in ['average', 'stack']
        self.ref_method = ref_method
        
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # image proj model
        self.image_proj_model = self.init_proj()
        self.image_proj_model_body = self.init_proj()

        self.load_ip_adapter()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor2_0_RP_Double_Adapter(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    height=self.height,
                    width=self.width,
                ).to(self.device, dtype=torch.float16)
                
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        # Load face adapter model
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Load body adapter model
        state_dict = torch.load(self.ip_ckpt_body, map_location="cpu")
        state_dict_to_kv_body = {}

        for key, value in state_dict["ip_adapter"].items():
            if 'to_k_ip' in key:
                state_dict_to_kv_body[key.replace('to_k_ip', 'to_k_ip_body')] = value
            if 'to_v_ip' in key:
                state_dict_to_kv_body[key.replace('to_v_ip', 'to_v_ip_body')] = value

        self.image_proj_model_body.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict_to_kv_body, strict=False)

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_all_seed(self, seed=42):
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

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor2_0_RP_Double_Adapter):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_image_embeds(self, pil_image, image_proj_model):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_image_embeds_average(self, pil_image):
        """
        pil_image: [[img1, img2], [img1, img2, img3]]
        """
        assert isinstance(pil_image, list) and isinstance(pil_image[0], list)
        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image_character in pil_image:
            clip_image = self.clip_image_processor(images=pil_image_character, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
            image_prompt_embeds_list.append(image_prompt_embeds.mean(dim=0 ,keepdim=True))
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds.mean(dim=0 ,keepdim=True))

        return torch.cat(image_prompt_embeds_list, dim=0), torch.cat(uncond_image_prompt_embeds_list, dim=0)

    def get_image_embeds_stack(self, pil_image):
        """
        pil_image: [[img_face1, img_cloth1], [img_face1, img_face2, img_cloth2], [bg]] 
        即 len(pil_image) = character的数量 + 1(背景)
            len(pil_image[0]) = character face ref img数量 + 1(衣服im)
        """
        assert isinstance(pil_image, list) and isinstance(pil_image[0], list)
        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        max_faces_num = max([len(pil_image_character) for pil_image_character in pil_image]) - 1

        # Character 先处理Character，再处理背景
        for pil_image_character in pil_image[:-1]: 
            pil_image_character_face = pil_image_character[:-1] # Face
            pil_image_character_body = pil_image_character[-1] # Body

            if len(pil_image_character_face) < max_faces_num: # 先padding到face img的最大数量，以便于后续向量化计算
                for _ in range(max_faces_num - len(pil_image_character_face)):
                    pil_image_character_face.append(Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)))
            
            # Face
            clip_image = self.clip_image_processor(images=pil_image_character_face, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds_face = self.image_proj_model(clip_image_embeds).view(1, self.num_tokens * max_faces_num, -1)
            # Body 
            clip_image = self.clip_image_processor(images=pil_image_character_body, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds_body = self.image_proj_model_body(clip_image_embeds)
            image_prompt_embeds = torch.cat([image_prompt_embeds_face, image_prompt_embeds_body], dim=1)

            # Face
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros((max_faces_num, *clip_image.shape[1:]), device=image_prompt_embeds.device), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds_face = self.image_proj_model(uncond_clip_image_embeds).view(1, self.num_tokens * max_faces_num, -1)
            # Body
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros((1, *clip_image.shape[1:]), device=image_prompt_embeds.device), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds_body = self.image_proj_model_body(uncond_clip_image_embeds)
            uncond_image_prompt_embeds = torch.cat([uncond_image_prompt_embeds_face, uncond_image_prompt_embeds_body], dim=1)

            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        # Background 处理背景
        pil_image_bg = pil_image[-1]
        if len(pil_image_bg) < max_faces_num+1:
            for _ in range(max_faces_num+1 - len(pil_image_bg)):
                pil_image_bg.append(Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)))
        clip_image = self.clip_image_processor(images=pil_image_bg, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)

        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)

        image_prompt_embeds_list.append(image_prompt_embeds.view(1, self.num_tokens * (max_faces_num+1), -1))
        uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds.view(1, self.num_tokens * (max_faces_num+1), -1))

        return torch.cat(image_prompt_embeds_list, dim=0), torch.cat(uncond_image_prompt_embeds_list, dim=0)

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(prompt)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "Asian,poorly drawn hands,poorly drawn face,poorly drawn feet,poorly drawn eyes,extra limbs,disfigured,deformed,ugly body"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        st = time.time()
        if self.ref_method == 'stack': #此处我们只会用到这个
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_stack(pil_image)
        else:
            raise ValueError
        et = time.time()
        print('Image embedding time:', et - st)

        text_embeds_all = (None, None, None, None)
        text_embeds_seperate = (None, None, None, None)
        if prompt is not None:
            self.set_all_seed(seed)

            text_embeds_seperate = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            
            text_embeds_all = self.pipe.encode_prompt(
                kwargs.get('ori_prompt'),
                negative_prompt=negative_prompt[0],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )

            prompt_embeds = concat_text_image_embeds(
                text_embeds_seperate[0], image_prompt_embeds, num_samples)
            negative_prompt_embeds = concat_text_image_embeds(
                text_embeds_seperate[1], uncond_image_prompt_embeds, num_samples)

        pooled_prompt_embeds = text_embeds_all[2][[0],:].repeat(num_samples, 1)
        pooled_negative_prompt_embeds = text_embeds_all[3][[0],:].repeat(num_samples, 1)
        
        generator = get_generator(seed, self.device)

        st = time.time()
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            is_webui_flag=True,
            **kwargs,
        ).images
        print('SD pipeline time:', time.time() - st)
        
        return images


def concat_text_image_embeds(
        txt_: torch.Tensor,
        img_: torch.Tensor,
        n_samples_: int) -> torch.Tensor:
    assert txt_.shape[-1] == img_.shape[-1]
    dim_ = img_.shape[-1]
    ret_ = torch.cat([txt_, img_], dim=1).unsqueeze(0) # e.g.(1,3,77+3*16,2048)
    return ret_.repeat(n_samples_, 1, 1, 1).view(n_samples_, -1, dim_) # e.g.(4,3*125,2048)
