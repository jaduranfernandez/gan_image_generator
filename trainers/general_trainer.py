import math
import torch
import torch.nn as nn
import numpy as np
from data_manipulation.utility import prepare_device
import matplotlib.pyplot as plt


def draw_images(generator, epoch, input_length):
    seed = 12
    n_rows = 2
    n_cols = 3
    scale = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*scale, n_rows*scale))


    np.random.seed(seed)
    noise = torch.tensor(np.random.normal(0, 1, (n_rows*n_cols, input_length)), dtype=torch.float)

    generator.to(torch.device("cpu"))
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise)

    for row_id in range(n_rows):
        for col_id in range(n_cols):
            index = row_id * n_cols + col_id
            face_image = fake_images[index].detach()
            face_image = (face_image + 1)/2
            axs[row_id, col_id].imshow(face_image.permute((1,2,0)))
            axs[row_id, col_id].set_xticklabels([])
            axs[row_id, col_id].set_yticklabels([])
            #axs[0, 0].set_title('Axis [0, 0]')
    fig.savefig("results/prediction_{0}.jpg".format(epoch))




class Trainer:
    def __init__(self, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer):
        self.device = prepare_device()
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer



    def train(self, dataloader, input_length: int = 10, n_epochs = 1):
        
        #for i in range(n_iterations):
        for epoch in range(n_epochs):
            print("Epoch {0}".format(epoch))

            for batch, (X) in enumerate(dataloader):
                # zero the gradients on each iteration
                self.generator_optimizer.zero_grad()

                # Create noisy input for generator
                noise = torch.tensor(np.random.normal(0, 1, (dataloader.batch_size, input_length)), dtype=torch.float)
                noise = noise.to(self.device)
                fake_data = self.generator(noise)

                # Generate examples of real data
                true_data = X.to(self.device)            
                true_labels = torch.ones(dataloader.batch_size,1).to(self.device)


                # Train the generator
                # We invert the labels here and don't train the discriminator because we want the generator
                # to make things the discriminator classifies as true.
                generator_discriminator_out = self.discriminator(fake_data)
                generator_loss = self.generator_loss(generator_discriminator_out, true_labels)
                generator_loss.backward()
                self.generator_optimizer.step()

                # Train the discriminator on the true/generated data
                self.discriminator_optimizer.zero_grad()
                true_discriminator_out = self.discriminator(true_data)
                true_discriminator_loss = self.discriminator_loss(true_discriminator_out, true_labels)

                generator_discriminator_out = self.discriminator(fake_data.detach())
                generator_discriminator_loss = self.generator_loss(generator_discriminator_out, torch.zeros(dataloader.batch_size,1).to(self.device))
                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

            torch.save(self.generator, str.format("saved_models/generator_{0}.pth",epoch))
            draw_images(self.generator, epoch, input_length)
            self.generator.to(self.device)


