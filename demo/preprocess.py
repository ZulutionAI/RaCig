from insightface.app import FaceAnalysis
from rembg import remove
from PIL import Image
import numpy as np

def expand_bbox(img, bbox, expansion_factor=1.2):
    """
    Expand the bounding box by a given factor while ensuring it stays within the image boundaries.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Calculate width and height of the original bbox
    width = x2 - x1
    height = y2 - y1

    # Increase width and height by the expansion factor
    new_width = int(width * (expansion_factor))
    new_height = int(height * (expansion_factor))

    # Calculate new x1, y1, x2, y2 ensuring the bbox stays within image boundaries
    new_x1 = max(0, x1 - (new_width - width) // 2)
    new_y1 = max(0, y1 - (new_height - height) // 2)
    new_x2 = min(img.width, x2 + (new_width - width) // 2)
    new_y2 = min(img.height, y2 + (new_height - height) // 2)

    # Ensure new_x2 and new_y2 are not smaller than new_x1 and new_y1
    new_x2 = max(new_x1 + 1, new_x2)
    new_y2 = max(new_y1 + 1, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]

class Preprocessor:
    def __init__(self) -> None:
        self.app = FaceAnalysis(name='antelopev2', allowed_modules=['detection', 'recognition'], root='./insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)
    
    def preprocess(self, img:Image, num_face=3, expand_ratio=1.3, remove_bg=True):
        
        face_img_list = []
        if remove_bg:
            img_np = remove(np.array(img), bgcolor=[255,255,255,255])[:,:,:3]
            img = Image.fromarray(img_np)
        else:
            img_np = np.array(img)

        for i in range(num_face):
            faces = self.app.get(img_np)
            faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
            bbox = faces['bbox']
            expanded_bbox = expand_bbox(img, bbox, expansion_factor=expand_ratio**(i))
            face_img = img.crop(expanded_bbox)
            face_img_list.append(face_img)

            if i == num_face // 2:
                body_img = img_np # y2

        return face_img_list, body_img

if __name__ == '__main__':
    p = Preprocessor()
    img = Image.open('assets/character/character_emma_whole.jpeg')
    p.preprocess(img)