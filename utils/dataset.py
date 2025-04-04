import torch
from PIL import Image
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, image_paths, labels, transform = None):
        self.image_paths = [p.strip() for p in image_paths]
        self.labels = [(1 if int(label) == 1 else 0) for label in labels]       # 1 = males | 0 = females
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        image = image.to(torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label