import math

import torch
import torch.nn as nn
import numpy as np

class Trainer:
    def __init__(self, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer):
        self.generator = generator
        self.generator_loss = generator_loss
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.discriminator_loss = discriminator_loss
        self.discriminator_optimizer = discriminator_optimizer


    def train_general(self, train_data: np.ndarray = [], input_length: int = 10, batch_size: int = 16, n_iterations: int = 500):
        
        for i in range(n_iterations):
            # zero the gradients on each iteration
            self.generator_optimizer.zero_grad()

            # Create noisy input for generator
            noise = np.random.normal(0,1,[batch_size,input_length])
            fake_data = self.generator(noise)

            # Generate examples of real data
            idx = np.random.randint(low=0, high=train_data.shape[0],size=batch_size)
            true_data = train_data[idx]
            true_labels = np.ones(batch_size)
            true_labels = torch.tensor(true_labels).float()
            true_data = torch.tensor(true_data).float()


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
            generator_discriminator_loss = self.generator_loss(generator_discriminator_out, torch.zeros(batch_size))
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            self.discriminator_optimizer.step()



