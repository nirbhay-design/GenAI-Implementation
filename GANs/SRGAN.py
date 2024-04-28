import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
import torch.optim as optim 

class ConvResBLock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, 
                      out_channels = out_channels, 
                      kernel_size= kernel_size, 
                      stride=stride,
                      padding = padding,
                      bias = False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(in_channels = out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding = padding,
                      bias = False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return x + self.conv_res(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, upsample_factor):
        super().__init__()
        out_channels = in_channels * upsample_factor ** 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2)
        self.pixel_shuffle = nn.PixelShuffle(upsample_factor)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        # [B, C, H, W] -> [B, C * r^2, H, W] -> [B, C, H * r, W * r]
        return self.prelu(self.pixel_shuffle(self.conv(x)))
    
class Generator(nn.Module):
    def __init__(self, in_channels, num_res_blocks):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = 64,
                      kernel_size = 9,
                      stride = 1,
                      padding = 4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[
                ConvResBLock(64, 64, 3, 1) for _ in range(num_res_blocks)
            ]
        )

        self.after_res = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.upsample1 = UpsampleBlock(64, 3, 1, 2)
        self.upsample2 = UpsampleBlock(64, 3, 1, 2)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=9, stride = 1, padding = 4)

    def forward(self, x):
        initial_x = self.initial_conv(x)
        res_blocks_x = self.res_blocks(initial_x)
        after_res_x = self.after_res(res_blocks_x)
        res_x = initial_x + after_res_x
        upsample_x = self.upsample2(self.upsample1(res_x))
        final_x = self.final_conv(upsample_x)
        return final_x


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

    
if __name__ == "__main__":
    crb = Generator(3, num_res_blocks=16)
    tsr = torch.rand(1,3,224,224)
    out = crb(tsr)
    print(out.shape)