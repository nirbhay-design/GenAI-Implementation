import torch 
import torch.nn as nn
import torch.nn.functional as F 

class UNet(nn.Module):
    def __init__(self, T = 1000):
        pass 
    
    def forward(self, x, t):
        return x # a noise predicted 
