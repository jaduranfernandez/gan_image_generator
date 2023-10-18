import torch.nn as nn


class FaceGenerator(nn.Module):
    def __init__(self):
        super(FaceGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 1024*4*4),
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
