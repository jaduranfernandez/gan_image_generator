import math
import torch
import torch.nn as nn
import numpy as np
from data_manipulation.utility import prepare_device


class Trainer:
    def __init__(self, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer):
        self.device = prepare_device()
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer



    def train(self, dataloader, input_length: int = 10):
        
        #for i in range(n_iterations):
        for batch, (X) in enumerate(dataloader):
            print("Epoch {0}".format(batch))
            # zero the gradients on each iteration
            self.generator_optimizer.zero_grad()

            # Create noisy input for generator
            noise = torch.tensor(np.random.normal(0, 1, (dataloader.batch_size, input_length)), dtype=torch.float)
            fake_data = self.generator(noise)

            # Generate examples of real data
            true_data = X.to(self.device)
            true_labels = torch.ones(dataloader.batch_size,1)


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
            generator_discriminator_loss = self.generator_loss(generator_discriminator_out, torch.zeros(dataloader.batch_size,1))
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if batch==1 or batch%300 == 0:
                torch.save(self.generator, str.format("saved_models/generator_{0}.pth",batch))


