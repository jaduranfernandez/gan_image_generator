import matplotlib.pyplot as plt 
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class AnimeFacesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        if(transform==None):
            transform = transforms.Compose([
                transforms.CenterCrop(64),
                transforms.Resize(64, interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

