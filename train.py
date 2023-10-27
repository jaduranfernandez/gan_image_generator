import torch.nn as nn
import torch
from torch.utils.data import DataLoader 
from data_manipulation.faces import FacesDataset
from models.anime_faces import Generator, Discriminator, init_weights
from trainers.general_trainer import Trainer


# Configuration variables
data_dir = 'data/anime_faces/'
batch_size = 32
epochs = 40
input_length = 100


print("Skere")

# Load data
dataset = FacesDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)



generator = Generator(input_length)
discriminator = Discriminator()
generator_loss = nn.BCELoss()
discriminator_loss = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)


generator.apply(init_weights)
discriminator.apply(init_weights)


trainer = Trainer(generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer)
#trainer.train(dataloader, input_length, n_epochs= epochs)




