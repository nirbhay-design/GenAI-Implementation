import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
from torchvision.utils import save_image
from PIL import Image 
import matplotlib.pyplot as plt 
from unet import UNet

class Forward():
    def __init__(self, T):
        self.T = T
        self.alpha_bar = self.cosine_alpha_bar(T)
        plt.plot(self.alpha_bar.numpy())
        plt.xlabel("timestamp")
        plt.ylabel("alpha")
        plt.savefig("alphas.png")
        plt.close()

    def cosine_alpha_bar(self, timestamps):
        T = timestamps
        s = 0.008
        t = torch.arange(float(T + 1))
        alphas = torch.cos(((t / T + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alpha_bar = alphas / alphas[0]
        return alpha_bar

    def __call__(self, x_0, t):
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t = alpha_bar_t.reshape(-1,1)
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon 
        return x_t, epsilon 
         
if __name__ == "__main__":
    img_path = "flickr_dog_000011.jpg"
    img = Image.open(img_path).convert("RGB")
    img = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(transforms.ToTensor()(img))
    
    fwd = Forward(1000)

    imgs = []
    for t in [100, 300, 500, 700, 900, 999]:

        img_10, epsilon = fwd(img, t)
        img_10 = img_10.clamp(min = -1.0, max = 1.0)
        imgs.append(img_10.unsqueeze(0))
    
    imgs = torch.vstack(imgs)

    print(imgs.shape)

    save_image((imgs + 1) / 2, "dog.png")

