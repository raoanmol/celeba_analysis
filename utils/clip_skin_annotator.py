import os
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image

IMAGE_DIR = '../data/img_align_celeba/'
OUTPUT_FILE = '../data/clip_annotations/list_skin_color_celeba.txt'

MODEL_NAME = 'ViT-L-14-quickgelu'
PRETRAINED = 'openai'

CONFIDENCE_THRESHOLD = 0.33

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model, preprocess, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained = PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

skin_tone_labels = [
    'A photo of a white person.',                   # 0 = Light
    'A photo of a brown person.',                   # 1 = Medium
    'A photo of a black person.'                    # 2 = Dark
]

text_inputs = tokenizer(skin_tone_labels).to(device)
text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim = -1, keepdim = True)

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0).to(device)

confidence_scores = []

with open(OUTPUT_FILE, 'w') as f:
    for image_name in tqdm(image_files, desc = 'Preprocessing images'):
        image_path = os.path.join(IMAGE_DIR, image_name)

        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim = -1, keepdim = True)

        similarity = (image_features @ text_features.T).squeeze(0)

        probabilities = torch.nn.functional.softmax(similarity, dim = 0)

        best_index = torch.argmax(probabilities).item()
        best_confidence = probabilities[best_index].item()

        confidence_scores.append(best_confidence)

        if best_confidence < CONFIDENCE_THRESHOLD:
            best_index = 2

        f.write(f'{image_name} {best_index}\n')

np.save('../data/clip_annotations/confidence_scores.npy', confidence_scores)
