import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import SiglipProcessor, SiglipModel

IMAGE_DIR = '../data/img_align_celeba/'
OUTPUT_FILE = '../data/siglip_annotations/list_skin_color_celeba.txt'
CONFIDENCE_THRESHOLD = 0.33

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "google/siglip-so400m-patch14-384"
processor = SiglipProcessor.from_pretrained(model_name)
model = SiglipModel.from_pretrained(model_name).to(device)
model.eval()

skin_tone_labels = [
    'A photo of a white person.',   # 0 = Light
    'A photo of a brown person.',   # 1 = Medium
    'A photo of a black person.'    # 2 = Dark
]

with torch.no_grad():
    inputs = processor(text=skin_tone_labels, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
confidence_scores = []

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return processor(images=image, return_tensors="pt")['pixel_values'].to(device)

with open(OUTPUT_FILE, 'w') as f:
    for image_name in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(IMAGE_DIR, image_name)
        image_tensor = preprocess_image(image_path)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(image_features, text_features.T).squeeze(0)
        probabilities = torch.nn.functional.softmax(similarity, dim=0)

        best_index = torch.argmax(probabilities).item()
        best_confidence = probabilities[best_index].item()
        confidence_scores.append(best_confidence)

        if best_confidence < CONFIDENCE_THRESHOLD:
            best_index = 2

        f.write(f'{image_name} {best_index}\n')

np.save('../data/siglip_annotations/confidence_scores.npy', confidence_scores)
