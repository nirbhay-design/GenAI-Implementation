import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision 
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        # x, y shape = [N,C,H,W]
        mx, stdx = self.get_mean_std(x)
        my, stdy = self.get_mean_std(y)

        nx = torch.div(x - mx, stdx)
        sy = stdy * nx + my
        return sy
    
    def get_mean_std(self, x):
        mx = torch.mean(x, dim = [0,2,3])
        stdx = torch.std(x, dim = [0,2,3])

        mx = mx.reshape(1, -1, 1, 1)
        stdx = stdx.reshape(1, -1, 1, 1)

        return mx, stdx 

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, n_layers):
        super().__init__()

        self.mapping_layers = nn.Sequential(
            *[nn.Linear(latent_dim, latent_dim) for _ in range(n_layers)]
        )
    
    def forward(self, x):
        return self.mapping_layers(x) 
    

class StyleGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x 
    

if __name__ == "__main__":
    adain = AdaIN()
    x = torch.rand(2,3,24,24)
    y = torch.rand(2,3,23,23)
    mapping_net = MappingNetwork(latent_dim=512, n_layers=8)
    print(mapping_net)
    print(mapping_net(torch.randn(2,512)).shape)

