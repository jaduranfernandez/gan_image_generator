import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


input_lenght = 4
fake_image = np.random.random(input_lenght)
discriminator = Discriminator(input_lenght)
discriminator(torch.tensor(fake_image))




