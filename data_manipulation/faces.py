import matplotlib.pyplot as plt 
import numpy as np
from keras import layers, Sequential
import torch
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        if(transform==None):
            transform = transforms.Compose([
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




def create_dataset(folder_path, file_name, data_path, first_image, last_image):
    
    dimensions = [128,128,3]
    values = np.zeros((last_image - first_image+1, dimensions[0],dimensions[1],dimensions[2]))
    # Resizing and rescaling
    resize_and_scale = Sequential()
    resize_and_scale.add(layers.Resizing(dimensions[0], dimensions[1]))
    resize_and_scale.add(layers.Rescaling(1./255))


    for it in range(first_image, last_image+1):
        filename = str(it).zfill(6) + ".jpg"
        image = plt.imread(data_path + filename)
        new_image = resize_and_scale(image)
        new_image = (2*new_image) - 1
        values[it-1,:] = new_image
    np.save(folder_path + file_name, values)

