import matplotlib.pyplot as plt 
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        if(transform==None):
            transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.Resize((128, 128)),  # Resize images to 128x128
                transforms.ToTensor(),  # Convert images to tensors
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

