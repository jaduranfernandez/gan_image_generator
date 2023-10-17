from base_model import BaseModel
import torch.nn as nn # basic building block for neural neteorks


class SimpleDiscriminator(BaseModel):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()

        #self.model_name = filename
        self.features_output_dim = 4*4*1024


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
            nn.Linear(self.features_output_dim, 1),            
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.features_output_dim)
        x = self.classifier(x)
        return x
    


class SimpleGenerator(BaseModel):
    def __init__(self):
        super(SimpleGenerator, self).__init__()

        #self.model_name = filename
        self.features_output_dim = 4*4*1024


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
            nn.Linear(self.features_output_dim, 1),            
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.features_output_dim)
        x = self.classifier(x)
        return x