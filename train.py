import os
import argparse
from pathlib import Path
import itertools
import time
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from torch.utils.tensorboard import SummaryWriter
from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor_mask import IPAttnProcessor2_0_RP_Double_Adapter as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    raise ValueError
from racig.pipeline import RaCigPipeline
from accelerate import DistributedDataParallelKwargs
import matplotlib.pyplot as plt
from dataset.transforms import (
    get_train_transforms_with_segmap,
    get_train_transforms_with_segmap_bbox,
    get_object_transforms,
    get_object_processor,
)
from dataset.data import get_data_loader, MyDataset
from racig.character import Character

def get_center(mask):
    """计算mask的中心点"""
    coords = torch.nonzero(mask)
    center = coords.float().mean(dim=0)
    return center

def assign_by_nearest(mask0, mask1, intersection):
    """根据最近邻方法将交集部分分配给mask0或mask1"""
    center0 = get_center(mask0)
    center1 = get_center(mask1)
    
    coords_intersection = torch.nonzero(intersection)
    dist0 = torch.norm(coords_intersection.float() - center0, dim=1)
    dist1 = torch.norm(coords_intersection.float() - center1, dim=1)
    
    mask0_assign = dist0 < dist1
    mask1_assign = ~mask0_assign
    
    mask0[coords_intersection[mask0_assign][:, 0], coords_intersection[mask0_assign][:, 1], coords_intersection[mask0_assign][:, 2]] = 1
    mask1[coords_intersection[mask1_assign][:, 0], coords_intersection[mask1_assign][:, 1], coords_intersection[mask1_assign][:, 2]] = 1
    
    return mask0, mask1

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

def pil_images_to_matplotlib_figure(pil_images):
    fig, axes = plt.subplots(1, len(pil_images), figsize=(len(pil_images) * 2, 3), dpi=300)
    if len(pil_images) == 1:
        axes = [axes]  # 如果只有一个图像，确保axes是一个列表
    for ax, img in zip(axes, pil_images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    return fig

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(
            self, 
            unet, 
            controlnet,
            image_proj_model,
            image_proj_model_body,
            adapter_modules, 
            ckpt_path=None,
            ckpt_body_path=None,
            num_tokens=16
            ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.image_proj_model = image_proj_model
        self.image_proj_model_body = image_proj_model_body
        self.ip_adapter = adapter_modules
        self.num_tokens = num_tokens

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path, ckpt_body_path)

    def forward(self, 
                noisy_latents, 
                timesteps, 
                encoder_hidden_states, # Text encoder hidden states: (batch_size, seq_len, hidden_dim)
                unet_added_cond_kwargs, 
                image_embeds_face,
                image_embeds_body, # Image embeds: (batch_size, max_num_objects, 257, hidden_dim)
                cond_img,
                segments):
        
        batch_size, max_num_objects, img_seq_len, hidden_dim = image_embeds_face.shape
        image_embeds_face = image_embeds_face.view(batch_size*max_num_objects, img_seq_len, hidden_dim)
        image_embeds_body = image_embeds_body.view(batch_size*max_num_objects, img_seq_len, hidden_dim)

        # Add body
        ip_tokens_face = self.image_proj_model(image_embeds_face) #
        ip_tokens_body = self.image_proj_model_body(image_embeds_body) #

        ip_tokens_face = ip_tokens_face.view(batch_size, max_num_objects*self.num_tokens, -1)
        ip_tokens_body = ip_tokens_body.view(batch_size, max_num_objects*self.num_tokens, -1)

        ip_tokens_face_list = ip_tokens_face.chunk(2, dim=1)
        ip_tokens_body_list = ip_tokens_body.chunk(2, dim=1)

        ip_tokens_person_0 = torch.cat([ip_tokens_face_list[0], ip_tokens_face_list[0], ip_tokens_body_list[0]] ,dim=1)
        ip_tokens_person_1 = torch.cat([ip_tokens_face_list[1], ip_tokens_face_list[1], ip_tokens_body_list[1]] ,dim=1)

        # Add face+body, bg
        encoder_hidden_states_all = torch.cat(
            [encoder_hidden_states, ip_tokens_person_0, 
             encoder_hidden_states, ip_tokens_person_1, 
             encoder_hidden_states, ip_tokens_person_0], 
             dim=1) # (batch_size, seq_len+num_tokens*max_num_objects, hidden_dim)
        
        if self.controlnet is not None:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states[:,:77,:],
                added_cond_kwargs=unet_added_cond_kwargs,
                controlnet_cond=cond_img,
                return_dict=False,
            )
            # Predict the noise residual
            noise_pred = self.unet(
                noisy_latents,
                timesteps, 
                encoder_hidden_states=encoder_hidden_states_all, 
                added_cond_kwargs=unet_added_cond_kwargs,
                down_block_additional_residuals=[
                            sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                        ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                cross_attention_kwargs = {"masks":segments, "use_bg_prompt":None, "use_bg_img":None, "multi_ref_num":[2,2,1]},
                return_dict=False,
            )[0]
        else:
            noise_pred = self.unet(
                noisy_latents,
                timesteps, 
                encoder_hidden_states=encoder_hidden_states, 
                added_cond_kwargs=unet_added_cond_kwargs,
                return_dict=False,
            )[0]
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str, ckpt_body_path: str):
        # Load face
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.ip_adapter.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Load body
        state_dict = torch.load(ckpt_body_path, map_location="cpu")
        state_dict_to_kv_body = {}

        for key, value in state_dict["ip_adapter"].items():
            if 'to_k_ip' in key:
                state_dict_to_kv_body[key.replace('to_k_ip', 'to_k_ip_body')] = value
            if 'to_v_ip' in key:
                state_dict_to_kv_body[key.replace('to_v_ip', 'to_v_ip_body')] = value
        
        self.image_proj_model_body.load_state_dict(state_dict["image_proj"])
        self.ip_adapter.load_state_dict(state_dict_to_kv_body, strict=False)

        print(f"Successfully loaded weights from checkpoint {ckpt_path}, {ckpt_body_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--train_height",
        type=int,
        default=1344,
    )

    parser.add_argument(
        "--train_width",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    parser.add_argument(
        "--num_image_tokens",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_num_objects",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--object_resolution",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--min_num_objects",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--balance_num_objects",
        action="store_true",
    )

    parser.add_argument(
        "--text_only_prob",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--uncondition_prob",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument("--multi", action="store_true", help="Use multi place injection if subject appears many times")
    parser.add_argument("--add_id_embeds", action="store_true")

    # Controlnet relevant args
    parser.add_argument("--train_controlnet", action="store_true")
    parser.add_argument("--use_controlnet", action="store_true")
    parser.add_argument("--control_cond", type=str, default="skeleton", choices=["skeleton", "bbox_img"])
    parser.add_argument("--cut_dataset", action="store_true")
    parser.add_argument("--use_origin_crop", action="store_true")
    parser.add_argument(
        "--no_object_augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--object_background_processor",
        type=str,
        default="random",
    )

    parser.add_argument(
        "--object_appear_prob",
        type=float,
        default=1,
    )
    
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
    )

    parser.add_argument(
        "--unet_backbone",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--val_steps",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    return args

@torch.no_grad()
def validate(model, vae, accelerator):
    pipe = RaCigPipeline(
        retrieval_model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        index_path="data/retrieve_info/features/fusion",
        json_dict_path="data/retrieve_info/index/fusion",
        data_root="data/Reelshot_retrieval",
        ip_ckpt="models/ipa_weights/ip-adapter-plus-face_sdxl_vit-h.bin",
        ip_ckpt_body="models/ipa_weights/ip-adapter-plus_sdxl_vit-h.bin",
        multi_ref_method='stack',
        controlnet_path="models/controlnet/model.safetensors",
        seperate_ref=True,
    )

    pipe.ip_model.pipe.vae = vae
    pipe.ip_model.pipe.controlnet = model.controlnet
    # Add the face reference for character1
    image1 = []
    image1.append(Image.open("assets/emmaface1.png"))
    image1.append(Image.open("assets/emmaface2.png"))
    # Add the clothes reference for character1
    image1.append(Image.open('assets/emmawhole.png'))

    # Add face and clothes reference for character2
    # 默认最后一个img是衣服，其余的img都是face：一个character可以有多个face ref
    image2 = []
    image2.append(Image.open("assets/brandtface1.png"))
    image2.append(Image.open("assets/brandtface2.png"))

    # Add the clothes reference for character2
    image2.append(Image.open("assets/brandtwhole.png"))

    character_woman = Character(name="emma", gender="1female", ref_img=image1)

    character_man = Character(name="brandt", gender="1male", ref_img=image2)

    prompt = "2characters, Emma hugging Brandt with eyes closed, photo, realistic\nEmma, (1female:1.5), girl next door, sweet, country girl, honey blond wavy hair with soft bangs, floral dress\nBrandt, (1man:1.5), soldier, 22yo, human, light brown short hair, side part, camouflage military uniform"
    bg_prompt = "night, Lounge bar, luxurious"
    bg_img = None

    with torch.autocast(accelerator.device.type):
        images, retrieval_img, pose_img, masks_visualize, idx, json_path = pipe(
            [character_woman, character_man],
            # [character_woman],
            prompt=prompt,
            bg_prompt=bg_prompt,
            bg_img=bg_img,
            num_samples=4,
            top_k=10,
            sample_method='index',
            skeleton_id=0,
            num_inference_steps=7,
            guidance_scale=2.5,
            ref_scale=1.0,
            seed=42,
        )

    return images, retrieval_img, pose_img, masks_visualize, idx


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        tbwriter = SummaryWriter(log_dir=os.path.join(args.logging_dir, 'visualize'))
    # Load scheduler, tokenizer and models.

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    if args.unet_backbone !=  "stabilityai/stable-diffusion-xl-base-1.0":
        unet = UNet2DConditionModel.from_single_file(args.unet_backbone, subfolder="unet")
    else:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    clip_image_processor = CLIPImageProcessor()
    if args.use_controlnet:
        controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-openpose-sdxl-1.0")

    # freeze parameters of models to save more memory
    if args.train_controlnet:
        controlnet.requires_grad_(True)
    else:
        if args.use_controlnet:
            controlnet.requires_grad_(False)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #ip-adapter-plus
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    image_proj_model_body = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_k_ip_body.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip_body.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size, 
                cross_attention_dim=cross_attention_dim, 
                num_tokens=args.num_tokens,
                height=args.train_height,
                width=args.train_width,
                )
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    model = IPAdapter(
        unet, 
        controlnet if args.use_controlnet else None,
        image_proj_model, 
        image_proj_model_body,
        adapter_modules, 
        args.pretrained_ip_adapter_path,
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
        )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model.weight_dtype = weight_dtype

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    model.requires_grad_(False)
    model.unet.requires_grad_(False)
    model.image_proj_model.requires_grad_(False)
    model.image_proj_model_body.requires_grad_(False)
    model.ip_adapter.requires_grad_(False)

    for name, module in model.controlnet.named_modules():
        if 'mid_block' in name or 'controlnet_down_blocks' in name:
            module.requires_grad_(True)
        else:
            module.requires_grad_(False)

    controlnet_params = list([p for p in model.controlnet.parameters() if p.requires_grad])
    params_to_opt = itertools.chain(
        controlnet_params if args.train_controlnet else []
        )
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    if accelerator.is_main_process:
        param_status = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                status = f"可训练参数: {name}"
                print(status)
            else:
                status = f"冻结参数: {name}"
                print(status)
            param_status.append(status)
        
        # 将参数状态保存到输出目录
        output_file = os.path.join(args.output_dir, "parameter_status.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(param_status))
        print(f"参数状态已保存到: {output_file}")
    # dataloader
    train_transforms = get_train_transforms_with_segmap_bbox(args) if args.use_controlnet else get_train_transforms_with_segmap(args)
    object_transforms_body = get_object_transforms(args, crop=False)
    object_transforms_face = get_object_transforms(args, crop=True)
    object_processor = get_object_processor(args)

    train_dataset = MyDataset(
        args.dataset_name,
        tokenizer,
        train_transforms,
        object_transforms_face,
        object_transforms_body,
        object_processor,
        device=accelerator.device,
        max_num_objects=args.max_num_objects,
        num_image_tokens=args.num_image_tokens,
        object_appear_prob=args.object_appear_prob,
        uncondition_prob=args.uncondition_prob,
        text_only_prob=args.text_only_prob,
        object_types=["people", "person"],
        split="train",
        min_num_objects=args.min_num_objects,
        balance_num_objects=args.balance_num_objects,
        args=args
    )
    train_dataloader = get_data_loader(train_dataset, args.train_batch_size)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin

            if accelerator.local_process_index == 0 and accelerator.is_main_process and global_step % args.val_steps == 0 and accelerator.is_local_main_process:
                returns = validate(accelerator.unwrap_model(model), vae, accelerator)
                figure = pil_images_to_matplotlib_figure(returns[0])
                tbwriter.add_figure(f'validation', figure, global_step=global_step)

            with accelerator.accumulate(model):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                object_pixel_values_body = batch["object_pixel_values_body"] # (batch_size, max_num_objects, 3, h, w)
                object_pixel_values_face = batch["object_pixel_values_face"]
                cond_img = batch[args.control_cond]
                batch_size, max_num_objects, channel, h, w = object_pixel_values_face.shape

                object_pixel_values_face = object_pixel_values_face.view(batch_size*max_num_objects, channel, h, w)
                object_pixel_values_body = object_pixel_values_body.view(batch_size*max_num_objects, channel, h, w)

                with torch.no_grad():
                    object_pixel_values_face = clip_image_processor(images=object_pixel_values_face, return_tensors="pt").pixel_values # TODO: 检查格式和大小
                    image_embeds_face = image_encoder(object_pixel_values_face.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    img_token_len, img_hidden_dim = image_embeds_face.shape[1:]
                    image_embeds_face = image_embeds_face.view(batch_size, max_num_objects, img_token_len, img_hidden_dim)

                    object_pixel_values_body = clip_image_processor(images=object_pixel_values_body, return_tensors="pt").pixel_values # TODO: 检查格式和大小
                    image_embeds_body= image_encoder(object_pixel_values_body.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    img_token_len, img_hidden_dim = image_embeds_body.shape[1:]
                    image_embeds_body = image_embeds_body.view(batch_size, max_num_objects, img_token_len, img_hidden_dim)

                with torch.no_grad():
                    encoder_output = text_encoder(batch['input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['input_ids'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat

                add_time_ids = [
                    torch.tensor([args.train_height,args.train_width]).unsqueeze(0).repeat(batch_size,1).to(accelerator.device),
                    torch.tensor([0,0]).unsqueeze(0).repeat(batch_size,1).to(accelerator.device),
                    torch.tensor([args.train_height,args.train_width]).unsqueeze(0).repeat(batch_size,1).to(accelerator.device),
                ]

                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                object_seg_map = batch['object_segmaps'] # (batch_size, h, w)
                object_face_seg_map = batch['object_face_segmaps']
                object_head_seg_map = batch['object_head_segmaps']
                num_objects = batch['num_objects']

                segments = []
                for i in range(len(num_objects)): # batch_dim
                    mask_seperate = []
                    for mask_id in range(num_objects[i]):
                        mask = object_seg_map[i, mask_id, ...]
                        face_mask = object_face_seg_map[i, mask_id, ...]
                        head_mask = object_head_seg_map[i, mask_id, ...]
                        head_face_mask = (face_mask + head_mask) > 0
                        mask_head_face = mask * head_face_mask
                        mask_body = mask * (~head_face_mask)
                        mask = torch.stack([mask_head_face, mask_body], dim=0) # (2, h, w)
                        mask_seperate.append(mask)

                    if num_objects[i] < max_num_objects:
                        for _ in range(max_num_objects - num_objects[i]):
                            mask_seperate.append(torch.zeros_like(mask, device=mask.device, dtype=mask.dtype))
                    mask_seperate = torch.stack(mask_seperate, dim=0)
                    segments.append(mask_seperate)
                segments = torch.stack(segments, dim=0) # (batch_size, max_num_objects, 2, h, w)
                segments = segments.permute(1,2,0,3,4) # (max_num_objects, 2, batch_size, h, w)

                noise_pred = model(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds_face, image_embeds_body, cond_img, segments)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                if accelerator.local_process_index == 0 and accelerator.is_main_process and global_step % 100 == 0 and accelerator.is_local_main_process:
                    to_pil = transforms.ToPILImage()

                    pil_images = [
                        to_pil(batch["object_pixel_values_body"][0,0,:,:,:]),
                        to_pil(batch["object_pixel_values_body"][0,1,:,:,:]),
                        to_pil(batch["object_pixel_values_face"][0,0,:,:,:]),
                        to_pil(batch["object_pixel_values_face"][0,1,:,:,:]),
                        to_pil(batch["segmap"][0].unsqueeze(0).repeat(3,1,1)),
                        to_pil(segments[0,0,0,:,:]), 
                        to_pil(segments[0,1,0,:,:]),
                        to_pil(segments[1,0,0,:,:]),
                        to_pil(segments[1,1,0,:,:]),
                        to_pil((batch["pixel_values"][0,:,:,:] + 1)/2)
                    ]
                    figure = pil_images_to_matplotlib_figure(pil_images)
                    tbwriter.add_figure(f'batch', figure, global_step=global_step)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
                    
            global_step += 1
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                # ip_ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}", f"ip_adapter.bin")
                # if accelerator.local_process_index == 0 and accelerator.is_main_process and accelerator.is_local_main_process:
                #     img_proj_state_dict = accelerator.unwrap_model(model).image_proj_model.state_dict()
                #     ip_adapter_state_dict = accelerator.unwrap_model(model).ip_adapter.state_dict()
                #     controlnet_state_dict = accelerator.unwrap_model(model).controlnet.state_dict()
                #     combined_state_dict = {
                #         'controlnet': controlnet_state_dict,
                #         'image_proj': img_proj_state_dict,
                #         'ip_adapter': ip_adapter_state_dict
                #     }
                #     torch.save(combined_state_dict, ip_ckpt)

                accelerator.save_state(save_path)

                for file in os.listdir(args.output_dir):
                    if file.startswith("checkpoint") and file != os.path.basename(save_path):
                        ckpt_num = int(file.split("-")[1])
                        if (args.save_steps is None or ckpt_num % args.save_steps != 0):
                            print(f"Removing {file}")
                            shutil.rmtree(os.path.join(args.output_dir, file))

            begin = time.perf_counter()
            if global_step >= args.max_train_steps:
                break

if __name__ == "__main__":
    main()    