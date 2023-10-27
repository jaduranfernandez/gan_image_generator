import torch.nn as nn


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, input_length = 100):
        super(Generator, self).__init__()
        
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_length, out_channels=512, kernel_size=4, stride=1, padding=0), # inputx1x1 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1), # 512x4x4 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), # 256x8x8 -> 128x16x16 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), # 128x16x16 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1), # 64x32x32 -> 3x64x64
            nn.Tanh()
        )
        

    def forward(self, x):
        x = self.deconv_block(x)
        return x




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1), # 3x64x64 -> 64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), # 64x32x32 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), # 128x16x16 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), # 256x8x8 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=0), # 512x4x4 -> 1x1x1
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x
