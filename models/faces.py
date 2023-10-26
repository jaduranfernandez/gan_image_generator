import torch.nn as nn


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class FaceGenerator(nn.Module):
    def __init__(self, input_length = 100):
        super(FaceGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_length, 1024*4*4),
            nn.BatchNorm1d(1024*4*4),
            nn.LeakyReLU(negative_slope=0.2)            
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1), # 8x8x512
            nn.BatchNorm2d(num_features=512,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1), # 16x16x256
            nn.BatchNorm2d(num_features=256,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), # 32x32x128
            nn.BatchNorm2d(num_features=128,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # 64x64x64
            nn.BatchNorm2d(num_features=64,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1), # 128x128x3
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.deconv_layers(x)
        return x


class FaceGeneratorAlt(nn.Module):
    def __init__(self, input_length = 100):
        super(FaceGeneratorAlt, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_length, 1024*4*4),
            nn.BatchNorm1d(1024*4*4),
            nn.LeakyReLU(negative_slope=0.2)            
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=1, padding=0), # 8x8x512
            nn.BatchNorm2d(num_features=512,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 16x16x256
            nn.BatchNorm2d(num_features=256,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 32x32x128
            nn.BatchNorm2d(num_features=128,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1), # 64x64x64
            nn.BatchNorm2d(num_features=64,momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=1), # 128x128x3
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.deconv_layers(x)
        return x




class FaceDiscriminator(nn.Module):
    def __init__(self):
        super(FaceDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2), # 64x64x64
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # 32x32x128
            nn.BatchNorm2d(num_features=128, momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 16x16x256
            nn.BatchNorm2d(num_features=256, momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2), # 8x8x512
            nn.BatchNorm2d(num_features=512, momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2), # 4x4x1024
            nn.BatchNorm2d(num_features=1024, momentum=0.3),
            nn.LeakyReLU(negative_slope=0.2),
        
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*1024, 1),            
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4*4*1024)
        x = self.classifier(x)
        return x
