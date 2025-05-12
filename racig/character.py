from PIL import Image

class Character:
    def __init__(self, name: str =None, gender: str =None, ref_img: Image =None, prompt: str =None, converted_prompt: str =None, sigma_id: int =None):
        self.name = name
        self.gender = gender
        self.ref_img = ref_img
        self.prompt = prompt
        self.converted_prompt = converted_prompt
        self.sigma_id = sigma_id