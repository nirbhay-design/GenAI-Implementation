import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
import torch.optim as optim 

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()

class Generator(nn.Module):
    def __init__(self, z_dim, channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        B, zdim = x.shape
        x = x.reshape(B, zdim, 1, 1)
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).flatten(1)
    
def essentials():
    netG = Generator(100, 1)
    netD = Discriminator(1) 
    optG = optim.Adam(params = netG.parameters(), lr = 0.0002)
    optD = optim.Adam(params = netD.parameters(), lr = 0.0002)
    return netG, netD, optG, optD 

def train(netG, netD, optG, optD, dataloader, lossfunction, epochs, device):
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
            z = torch.randn(data.shape[0], 100).to(device)
            with torch.no_grad():
                fake_data = netG(z)
            disc_real = netD(data)
            disc_fake = netD(fake_data)

            real_labels = torch.ones_like(disc_real)
            fake_labels = torch.zeros_like(disc_fake)

            optD.zero_grad()

            loss_disc_real = lossfunction(disc_real, real_labels)
            loss_disc_fake = lossfunction(disc_fake, fake_labels)

            loss_d = loss_disc_fake + loss_disc_real
            loss_d.backward()

            optD.step()

            # generator loss min log(1 - D(G(z))) -> max log(D(G(z))) -> min -log(D(G(z)))
            dis_gen_fake = netD(fake_data)
            dis_gen_real_labels = torch.ones_like(dis_gen_fake)
            gen_loss = lossfunction(dis_gen_fake, dis_gen_real_labels)

            optG.zero_grad()
            gen_loss.backward()
            optG.step()

            genloss += gen_loss / (len_data * data.shape[0])
            discloss += loss_d / (len_data * data.shape[0])

            progress(idx+1, len_data)
        
        print(f"epoch: [{epoch}/{epochs}], genloss: {genloss:.3f}, discloss: {discloss:.3f}")

    return netG, netD

def dataset():
    transformations = transforms.Compose([
        transforms.Resize((64,64)), 
        transforms.ToTensor(), # makes the tensor between 0 to 1
        transforms.Normalize([0.5], [0.5])
    ])
    train_data = torchvision.datasets.MNIST('../dataset', train=True, download=True, transform = transformations)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    return train_loader

def main():
    netG, netD, optG, optD = essentials()
    dataloader = dataset()
    device = torch.device("cuda:0")
    lossfunction = nn.BCELoss()
    epochs = 20
    train(netG, netD, optG, optD, dataloader, lossfunction, epochs, device)


if __name__ == "__main__":
    main()