import os
import io
import gradio as gr
import json
import requests
import random
import base64
import string
from PIL import Image
from racig.pipeline import RaCigPipeline
from racig.character import Character
from demo.preprocess import Preprocessor
from datetime import datetime

class Gradio2IP:
    def __init__(self, default_path, pipe, server_name, port) -> None:
        self.pipe = pipe
        self.server_name = server_name
        self.port = port
        self.out_dir = 'demo/outs'
        self.preprocessor = Preprocessor()
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.uploaded_characters = []
        for dir in os.listdir(default_path):
            imgs = []
            clothes = []
            prompt = ''
            dirpath = os.path.join(default_path, dir)
            for file in os.listdir(dirpath):
                filepath = os.path.join(dirpath, file)
                if "face" in file and os.path.isdir(filepath):
                    imgs.extend([Image.open(os.path.join(filepath, face)) for face in os.listdir(filepath)])
                elif "body" in file and os.path.isdir(filepath):
                    clothes.extend([Image.open(os.path.join(filepath, body)) for body in os.listdir(filepath)])
                elif "prompt" in file:
                    with open(filepath) as prompt_file:
                        prompt = prompt_file.read().strip()
                elif not os.path.isdir(filepath):
                    imgs.append(Image.open(filepath))
            imgs.extend(clothes)
            name = dir.split('.')[0].split('_')[0]
            gender = dir.split('.')[0].split('_')[1]
            new_character = Character(
                name = name.lower(), gender=gender, ref_img=imgs, prompt=prompt
            )
            self.uploaded_characters.append(new_character)

        self.default_prompt = {
            'prompts': "2 characters, emma and brandt is holding hands walking on the street",
            'negative_prompts': "poorly drawn hands,poorly drawn face,poorly drawn feet,poorly drawn eyes ,extra limbs, disfigured,deformed,ugly body"
        }
        

    def add_character(self, images, body, gender, name, prompt, expand_ratio):
        if None in [gender, name]:
            gr.Warning("Please check your inputs and upload again.")
            return self.uploaded_characters, characterlist,  None, None, None
        
        input_faces = [images] if images is not None else []
        input_body = [body] if body is not None else []
        if len(input_faces)>0:
            character_face, character_body = self.preprocessor.preprocess(input_faces[0], num_face=2, expand_ratio=expand_ratio, remove_bg=False)
            new_ref_imgs = character_face + [Image.fromarray(character_body)]
            if len(input_body)>0: 
                new_ref_imgs = new_ref_imgs[:-1] + input_body
        else:
            new_ref_imgs = []
        
        uploaded = [index for index, character in enumerate(self.uploaded_characters) if character.name == name.lower() and character.gender == gender]
        if len(uploaded) == 1:
            if prompt != "":
                self.uploaded_characters[uploaded[0]].prompt = prompt
            if len(new_ref_imgs)>0: 
                self.uploaded_characters[uploaded[0]].ref_img = new_ref_imgs
            elif len(input_body)>0:
                self.uploaded_characters[uploaded[0]].ref_img = self.uploaded_characters[uploaded[0]].ref_img[:-1] + input_body
            characterlist = [[item.name, item.gender] for i, item in enumerate(self.uploaded_characters)]
            return self.uploaded_characters, characterlist,  None, None, self.uploaded_characters[uploaded[0]].ref_img, self.uploaded_characters[uploaded[0]].prompt
        
        new_character = Character(
            name = name.lower(), gender=gender, ref_img=new_ref_imgs, prompt=prompt
        )
        self.uploaded_characters.append(new_character)
        characterlist = [[item.name, item.gender] for i, item in enumerate(self.uploaded_characters)]
        return self.uploaded_characters, characterlist,  None, None, new_character.ref_img, new_character.prompt
    
    def show_ref(self, name, gender):
        uploaded = [index for index, character in enumerate(self.uploaded_characters) if character.name == name.lower() and character.gender == gender]
        if len(uploaded) == 1:
            return self.uploaded_characters[uploaded[0]].ref_img, self.uploaded_characters[uploaded[0]].prompt
        else:
            gr.Warning(f"Character {name}({gender}) not uploaded !)")
            return None, 'Character not found !'

    def clear(self, name, gender):
        uploaded = [character for character in self.uploaded_characters if character.name == name.lower() and character.gender == gender]
        if len(uploaded) == 1:
            self.uploaded_characters.remove(uploaded[0])
            return self.uploaded_characters, [[item.name, item.gender] for i, item in enumerate(self.uploaded_characters)], '', '', None, ''

        else:
            gr.Warning(f"Character {name}({gender}) not uploaded !)")
            return self.uploaded_characters, [[item.name, item.gender] for i, item in enumerate(self.uploaded_characters)], '', '', None, 'Character not found!'

    
    def racig(self, prompt, bg_prompt, bg_img, num_samples, num_steps, sample_method, top_k, sk_id, seed, guidance_scale, ref_scale, kp_conf, progress=gr.Progress(track_tqdm=True)):
        input_character = []
        punctuation = string.punctuation
        translation_table = str.maketrans(punctuation, ' ' * len(punctuation))
        prompt_no_punc = prompt.translate(translation_table)
        for character in self.uploaded_characters:
            name = character.name.split(' ')
            name_in = True
            for word in name:
                if word not in (prompt_no_punc.lower()).split(' '):
                    name_in = False
                    break
            if name_in:    
                input_character.append(character)
                prompt += f"\n{character.prompt}"
        if seed == -1:
            sd = random.randint(1, 1000000)
        else: sd = seed
        racig_images, retrieved_img, skeleton_img, masks_vis, skeleton_index, _ = self.pipe(
            character_list=input_character, prompt=prompt, bg_prompt=bg_prompt, bg_img=bg_img, num_samples=num_samples, top_k=top_k, skeleton_id=sk_id-1, sample_method=sample_method, num_inference_steps=num_steps, seed=sd, guidance_scale=guidance_scale, ref_scale=ref_scale, kp_conf=kp_conf
        )
        for idx, img in enumerate(racig_images[:num_samples]):
            img.save(f"{self.out_dir}/racig_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png")
        
        return racig_images[:num_samples], [retrieved_img, skeleton_img]+masks_vis, skeleton_index
        
    def demo(self):
        with gr.Blocks() as demo:
            uploaded = gr.State(value=self.uploaded_characters)
            with gr.Row():
                with gr.Column(scale=4):
  
                    with gr.Group():
                        gr.Markdown("# Characters \nUpload a new character or update characters' information for RaCig. Default characters have been uploaded. \n")
                        headers = ["Name", "Gender"]
                        character_list = gr.List(value=[[item.name, item.gender] for i, item in enumerate(uploaded.value)], row_count=(5, "dynamic"), headers=headers, col_count=2)
                        
                        refs = gr.Gallery(label="", elem_id="gallery_refs")
                        
                        gr.Markdown("## References")
                        with gr.Row():
                            name_input = gr.Textbox(label="Name")
                            gender_input = gr.Radio(choices=["1male", "1female"], label="Gender")
                        character_prompt = gr.Textbox(label="Prompt for Character", placeholder="Emma, (1female:1.5), girl next door, sweet, country girl, honey blond wavy hair with soft bangs, floral dress")
                        with gr.Row():
                            image_input = gr.Image(type="pil", label="Reference Images")
                            body_input = gr.Image(type="pil", label="Suits Reference")
                            
                        with gr.Accordion(label="Advanced Options", open=False):
                            expand_ratio_input = gr.Slider(label="Face Expand Ratio", minimum=1.0, maximum=3.0, step=0.1, value=2.0, info="Controls how much the face bounding box is expanded before processing.")

                        with gr.Row():
                            add_button = gr.Button("Upload",size="md", variant="primary")
                            vis_button = gr.Button("Visualize",size="md", variant="primary")
                            clear_button = gr.Button("Clear",size="md", variant="stop")  
                        do_clear = gr.Button("Do", visible=False, elem_id="do_clear")
                        cancel_clear = gr.Button("Cancel", visible=False, elem_id="cancel_clear")

                       
                    with gr.Group():
                        gr.Markdown("# Prompts \nCharacters mentioned in your prompts must be uploaded first!")
                        prompt_input = gr.Textbox(value=self.default_prompt["prompts"], label="prompts", )
                        negative_prompt_input = gr.Textbox(value=self.default_prompt["negative_prompts"], label='negative prompts')
                        with gr.Accordion(label="Background", open=True):
                            bg_input = gr.Textbox(value="in a luxurious bar", label="Background prompt")
                            bg_img = gr.Image(type="pil", label="Upload Background Image")
                

                            
                with gr.Column(scale=4):
                    with gr.Group():
                        gr.Markdown("# RaCig")
                        pipe_result = gr.Gallery(label="Generated Images from RaCig")
                        retrieved_sk = gr.Gallery(label="Skeleton and reference retrieved from prompts")
                    racig_button = gr.Button("Run RaCig", variant="primary")
                    with gr.Group():
                        gr.Markdown("Settings")
                        racig_seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=1000000,
                            step=1,
                            value=-1,
                            info="If set to -1, a different seed will be used each time.",
                        )
                        num_samples = gr.Slider(
                            label="Number of generated images",
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=4,
                        )
                        num_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=100,
                            step=5,
                            value=17,
                        )
                        guidance_sclae = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.5,
                            value=2.5,
                        )
                        ref_scale_input = gr.Slider(
                            label="Reference Scale",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0,
                        )

                        with gr.Accordion(label="Settings for skeleton retrieval", open=False):
                            sample_method = gr.Radio(value="random", choices=["random", "index"], label="Sample method")
                            top_k = gr.Slider(
                                label="Number of matched skeletons retrieved",
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=10,
                            )
                            skeleton_id = gr.Slider(
                                label="Skeleton index",
                                minimum=0,
                                maximum=19,
                                step=1,
                                value=0,
                                info="Index should be within the number of matched skeletons retrieved!"
                            )
                            kp_conf = gr.Slider(
                                label="Keypoints confidence",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.5,
                            )

            racig_inputs = [
                prompt_input, 
                bg_input, 
                bg_img, 
                num_samples, 
                num_steps, 
                sample_method, 
                top_k, 
                skeleton_id,
                racig_seed,
                guidance_sclae,
                ref_scale_input,
                kp_conf
            ]

            def update_inputs_on_select(evt: gr.SelectData):
                selected_row = evt.row_value
                return selected_row[0], selected_row[1]

            character_list.select(update_inputs_on_select, None, [name_input, gender_input])

            vis_button.click(self.show_ref, inputs=[name_input, gender_input], outputs=[refs, character_prompt])

            add_button.click(self.add_character, inputs=[image_input, body_input, gender_input, name_input, character_prompt, expand_ratio_input], outputs=[uploaded, character_list, image_input, body_input, refs, character_prompt])

            clear_button.click(None, None, None, js="""
                () => {
                    if (confirm('Are you sure you want to remove the character?')) {
                        document.getElementById('do_clear').click();
                    } else {
                        document.getElementById('cancel_clear').click();
                    }
                }
            """)
            do_clear.click(fn=self.clear, inputs=[name_input, gender_input], outputs=[uploaded, character_list, gender_input, name_input, refs, character_prompt])
            cancel_clear.click(None, None, None)
            
            racig_button.click(fn=self.racig, inputs=racig_inputs, outputs=[pipe_result, retrieved_sk, skeleton_id])

            demo.launch(
                show_error=True,
                server_name=self.server_name,
                server_port=self.port,
                share=True,
            )


if __name__ == "__main__":
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

    g = Gradio2IP('demo/default_multi_ref', pipe, "0.0.0.0", 8081)
    g.demo()