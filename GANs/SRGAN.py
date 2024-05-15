import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
import torch.optim as optim 
from utils import progress

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
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv_block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_dims, strides):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2)
        )

        num_conv_blocks = len(feature_dims)

        self.conv_blocks = nn.ModuleList()

        cur_channels = 64
        for i in range(num_conv_blocks):
            self.conv_blocks.append(
                ConvBlock(cur_channels, feature_dims[i], kernel_size=3, stride=strides[i])
            )
            cur_channels = feature_dims[i]

        self.aap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(1)
        self.linear_layers = nn.Sequential(
            nn.Linear(cur_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        for layers in self.conv_blocks:
            x1 = layers(x1)
        aap_x = self.flatten(self.aap(x1))
        linear_x = self.linear_layers(aap_x)

        return linear_x
    
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1").features[:36].eval()

        for params in self.vgg.parameters():
            params.requires_grad = True


        self.mse = nn.MSELoss()

    def forward(self, x1, x2):
        x1_features = self.vgg(x1)
        x2_features = self.vgg(x2)

        return self.mse(x1_features, x2_features)
    
class DiscLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, disc_real, disc_fake):
        real_labels = torch.ones_like(disc_real)
        fake_labels = torch.zeros_like(disc_fake)

        loss_disc_real = self.bce(disc_real, real_labels)
        loss_disc_fake = self.bce(disc_fake, fake_labels)

        loss_d = loss_disc_fake + loss_disc_real

        return loss_d

class GenLoss(nn.Module):
    def __init__(self, reconstruction='VGG'):
        super().__init__()
        self.bce = nn.BCELoss()
        if reconstruction == "VGG":
            self.rec = VGGLoss()
        else:
            self.rec = nn.MSELoss()

    def forward(self, dis_gen_fake, x1, x2):
        dis_gen_real_labels = torch.ones_like(dis_gen_fake)
        gen_rec = self.rec(x1, x2)
        loss_gen = self.bce(dis_gen_fake, dis_gen_real_labels) + gen_rec
        return loss_gen

    
def train(netG, netD, optG, optD, dataloader, loss_d, loss_g, epochs, device):
    netG.train()
    netD.train() 

    netG = netG.to(device)
    netD = netD.to(device)
    len_data = len(dataloader)

    for epoch in range(epochs):
        genloss = 0
        discloss = 0
        for idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            # discriminator loss min -(log(D(x)) + log(1 - D(G(z))))
            fake_data = netG(data)
            disc_real = netD(data)
            disc_fake = netD(fake_data.detach())

            loss_d = loss_d(disc_real, disc_fake)

            optD.zero_grad()
            loss_d.backward()
            optD.step()

            # generator loss min log(1 - D(G(z))) -> max log(D(G(z))) -> min -log(D(G(z)))
            dis_gen_fake = netD(fake_data)
            gen_loss = loss_g(dis_gen_fake, data, fake_data)

            optG.zero_grad()
            gen_loss.backward() 
            optG.step()

            genloss += (gen_loss / len_data)
            discloss += (loss_d / len_data)

            progress(idx+1, len_data, BGL=float(gen_loss), BDL=float(loss_d), GL = float(genloss), DL = float(discloss))
        
        print(f"epoch: [{epoch}/{epochs}], genloss: {genloss:.3f}, discloss: {discloss:.3f}")

    return netG, netD
    
if __name__ == "__main__":
    crb = Discriminator(3, [64,128,128,256,256,512,512], [2,1,2,1,2,1,2])
    tsr = torch.rand(1,3,224,224)
    out = crb(tsr)
    print(out.shape)