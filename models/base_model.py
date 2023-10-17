import torch.nn as nn # basic building block for neural neteorks
import torch


class BaseModel(nn.Module):
    """ 
    Template of a Neural Network in Pytorch
    """
    def __init__(self, model_folder = "trained_models"):
        """ Initialize the neural network """        
        super(BaseModel, self).__init__()
        self.model_name = "Base_model"
        self.model_folder = model_folder
        self.features = nn.Sequential()
        self.classifier = nn.Sequential()
        self.features_output_dim = 0


    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.features(x)
        x = x.view(-1, self.features_output_dim)
        x = self.classifier(x)
        return x


    def save_model(self):
        torch.save(self, self.model_folder + "/" + self.model_name + ".pth")