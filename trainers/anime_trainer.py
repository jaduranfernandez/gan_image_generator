import torch
import torch.nn as nn
import numpy as np
from data_manipulation.utility import prepare_device
import matplotlib.pyplot as plt
from tqdm import tqdm



def draw_images(generator, epoch, input_length):
    n_rows = 2
    n_cols = 3
    scale = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*scale, n_rows*scale))


    #noise = torch.tensor(np.random.normal(0, 1, (n_rows*n_cols, input_length)), dtype=torch.float)
    noise = torch.randn(n_rows*n_cols, input_length, 1, 1)

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
        for epoch in range(n_epochs):
            print("Epoch {0}".format(epoch))
            for real_images in tqdm(dataloader):
                real_images = real_images.to(self.device)


                # Pass real images through discriminator
                D_out_real = self.discriminator(real_images)
                label_real = torch.ones(D_out_real.shape).to(self.device)
                real_loss = self.discriminator_loss(D_out_real, label_real)


                # Generate fake images
                noise = torch.randn(dataloader.batch_size, input_length, 1, 1).to(self.device)
                fake_images = self.generator(noise)
                
                # Pass fake images through discriminator
                D_out_fake = self.discriminator(fake_images)
                label_fake = torch.ones(D_out_fake.shape).to(self.device)
                fake_loss = self.discriminator_loss(label_fake, D_out_fake) 


                loss_d = real_loss + fake_loss

                self.discriminator_optimizer.zero_grad()
                loss_d.backward(retain_graph = True)
                self.discriminator_optimizer.step()

                # Generate fake images
                noise2 = torch.randn(dataloader.batch_size, input_length, 1, 1).to(self.device)
                fake_images2 = self.generator(noise2)
                gen_steps = 1
                for i in range(0, gen_steps):
                # Try to fool the discriminator
                    D_out_fake2 = self.discriminator(fake_images2)
            
                    # The label is set to 1(real-like) to fool the discriminator
                    label_real1 = torch.full(D_out_fake2.shape, 1.0).to(self.device)
                    loss_g = self.generator_loss(label_real1, D_out_fake2)
                
                    # Update generator weights
                    self.generator_optimizer.zero_grad()
                    loss_g.backward(retain_graph = (i<gen_steps -1 ))
                    self.generator_optimizer.step()


            torch.save(self.generator, str.format("saved_models/generator_{0}.pth",epoch))
            draw_images(self.generator, epoch, input_length)
            self.generator.to(self.device)


