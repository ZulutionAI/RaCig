from racig.pipeline import RaCigPipeline
from racig.character import Character
from PIL import Image
import os

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

# Add the face reference for character1
image1 = []
image1.append(Image.open("demo/default_multi_ref/Emma_1female/face/woman_1_face.png"))
image1.append(Image.open("demo/default_multi_ref/Emma_1female/face/woman_2_face.png"))
# Add the clothes reference for character1
image1.append(Image.open('demo/default_multi_ref/Emma_1female/body/woman_2_body_.png'))


# Add face and clothes reference for character2
# 默认最后一个img是衣服，其余的img都是face：一个character可以有多个face ref
image2 = []
image2.append(Image.open("demo/default_multi_ref/Brandt_1male/face/man_1_face.png"))
image2.append(Image.open("demo/default_multi_ref/Brandt_1male/face/man_2_face.png"))
# Add the clothes reference for character2
image2.append(Image.open("demo/default_multi_ref/Brandt_1male/body/man_body.png"))

character_woman = Character(name="emma", gender="1female", ref_img=image1)

character_man = Character(name="brandt", gender="1male", ref_img=image2)

prompt = "2characters, emma is walking with brandt, hand in hand"
bg_prompt = "a city street"
bg_img = None

total_time = 0
test_nums = 1
os.makedirs('output', exist_ok=True)
for i in range(test_nums):
    racig_images, retrieved_img, skeleton_img, masks_vis, skeleton_index, _  = pipe(
        [character_woman, character_man],
        prompt=prompt,
        bg_prompt=bg_prompt,
        bg_img=bg_img,
        num_samples=1,
        top_k=10,
        sample_method='random',
        num_inference_steps=17,
        guidance_scale=2.5,
        ref_scale=1.0,
        seed=0
    )
    for id, img in enumerate(racig_images):
        img.save(f'output/img_{i:02}_{id:02}.png')
    retrieved_img.save(f'output/retrieved_{i:02}.png')
    skeleton_img.save(f'output/skeleton_{i:02}.png')
    # composed_img.save('wukong/retrieved.png')

