from PIL import Image
import random
import glob
import matplotlib.pyplot as plt
from data_manipulation.anime_faces import AnimeFacesDataset
from torch.utils.data import DataLoader 

link = "https://aihalapathirana.medium.com/generative-adversarial-networks-for-anime-face-generation-pytorch-1b4037930e21"

def draw_image(image):
    face_image = image.detach()
    face_image = (face_image + 1)/2
    plt.imshow(face_image.permute((1,2,0)))
    plt.show()


data_dir = 'data/anime_faces/'
dataset = AnimeFacesDataset(data_dir)
print(len(dataset))
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
draw_image(dataloader.dataset[0])
