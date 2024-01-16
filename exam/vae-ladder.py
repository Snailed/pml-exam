from __future__ import print_function
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid
import argparse
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Lightning')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--checkpoint-path', type=str, default='',
                    help='resume from checkpoint (default: None)')
args = parser.parse_args()


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Structure: x <- z1 <- z2
        # x is a cont. bernoulli, z1 is a gaussian, z2 is a gaussian
        self.encoder0_to_1 = nn.Sequential(nn.Linear(28 * 28, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 4), nn.Sigmoid())
        self.encoder1_to_2 = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 4))
        self.decoder2_to_1 = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 4))
        self.decoder1_to_0 = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 28 * 28), nn.Sigmoid())
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        z1 = self.encoder0_to_1(x.view(x.size(0), -1))
        mu, logvar = z1[:,:2], z1[:,2:4]
        z2 = self.encoder1_to_2(self.reparameterize(mu, logvar))
        return z2

    # Reconstruction + KL divergence losses summed over all elements and batch
    # Loss is ELBO 
    def loss_function(self, recon_x, x, mu, logvar):
        # KLD(q(Z|X), p(Z)) (both a gaussian, so closed-form expression)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # CBCE = E_q(Z|X) [ln p(X|Z)]
        CBCE = F.binary_cross_entropy(recon_x, x, reduction='sum') + torch.sum(torch.distributions.ContinuousBernoulli(probs=recon_x)._cont_bern_log_norm())
        return KLD + CBCE

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def ladder_mu(self, mu_q, mu_p, logvar_q, logvar_p):
        return (mu_q * (-2 * logvar_q).exp() + mu_p * (-2 * logvar_q).exp())/((-2*logvar_q).exp() + (-2*logvar_p).exp())

    def ladder_logvar(self, logvar_q, logvar_p):
        return torch.log(1 / ((-2*logvar_q).exp() + (-2*logvar_p).exp()))

    def split(self, x):
        return x[:,:2], x[:,2:4]
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        

        # Encode x to z1
        x_enc = self.encoder0_to_1(x.view(x.size(0), -1))
        z1_p_mu, z1_p_logvar = self.split(x_enc)
        z1_p_dist = torch.distributions.Normal(z1_p_mu, (0.5*z1_p_logvar).exp())
        z1 = z1_p_dist.rsample()

        # Encode z1 to z2
        z1_enc = self.encoder1_to_2(z1)
        z2_p_mu, z2_p_logvar = self.split(z1_enc)
        z2_p_dist = torch.distributions.Normal(z2_p_mu, (0.5*z2_p_logvar).exp())
        z2 = z2_p_dist.rsample()

        # Decode z2 to z1
        z2_dec = self.decoder2_to_1(z2)
        z1_q_mu_hat, z1_q_logvar_hat = self.split(z2_dec)
        z1_q_mu = z1_q_mu_hat
        z1_q_logvar = z1_q_logvar_hat
        z1_q_dist = torch.distributions.Normal(z1_p_mu, (0.5*z1_p_logvar).exp())
        # z1_q_mu = self.ladder_mu(z1_q_mu_hat, z1_p_mu, torch.log(1 + z1_q_logvar_hat.exp().exp()), z1_p_logvar)
        # z1_q_logvar = self.ladder_logvar(torch.log(1 + z1_q_logvar_hat.exp().exp()), z1_p_logvar)
        z1_q = z1_q_dist.rsample()

        # Decode from z1 to x
        z1_dec = self.decoder1_to_0(z1_q)
        recon_x = z1_dec
        
        z1_q_dist = torch.distributions.Normal(z1_q_mu, (0.5*z1_q_logvar).exp())
        #z1_p_dist = torch.distributions.Normal(z1_p_mu, (0.5*z1_p_logvar).exp())
        kl_z2_to_z1 = torch.sum(torch.distributions.kl_divergence(z1_q_dist, z2_p_dist))
        
        # Compute loss from 1 to 0
        loss = self.loss_function(recon_x, x.view(-1, 784), z2_p_mu, z2_p_logvar)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, test = random_split(dataset, [55000, 5000])

if args.checkpoint_path == '':
    model = VAE()
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model, DataLoader(train, num_workers=7, batch_size=128))
else:
    model = VAE.load_from_checkpoint(args.checkpoint_path)
    model.eval()

def show(imgs, uni_samples, x_samples, y_samples):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=x_samples, yticklabels=y_samples, xticks=np.arange(20)*30 + 14, yticks=np.arange(20)*30 + 16)
        axs[0,i].set_xticklabels(x_samples, rotation=90)
        axs[0,i].set_xlabel('$z_0$')
        axs[0,i].set_ylabel('$z_1$')
    return fig
         
def scatter_latent():
    x = torch.randn(1, 64)
    for x, y in DataLoader(test, batch_size=5000):
        class_points = {label: [] for label in set(list(y.numpy()))}
        points = model(x)
        # Separate points based on labels
        for point, label in zip(points, y):
            class_points[int(label.numpy())].append(point)
# Scatter plot with labels and legend
        for label, points in class_points.items():
            class_x = [p[0] for p in points]
            class_y = [p[1] for p in points]
            plt.scatter(class_x, class_y, label=f'Class {label}', s=10)
    # Add legend
    plt.legend()
    plt.show()

def recon_grid():
    k = 20
    x_uniform = torch.linspace(0.00001, 0.99999, k)
    y_uniform = torch.linspace(0.00001, 0.99999, k)
    x_normal = torch.distributions.normal.Normal(0, 1).icdf(x_uniform)
    y_normal = torch.distributions.normal.Normal(0, 1).icdf(y_uniform)
    grid_x, grid_y = torch.meshgrid(x_normal, y_normal, indexing='ij')

    #z1 = model.decoder2_to_1(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
     
    sample = model.decoder1_to_0(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
    grid = make_grid(sample.view(20 * 20, 1, 28, 28), nrow=20)
    show(grid, x_uniform.numpy(), x_normal.numpy(), y_normal.numpy())
    plt.title('Visualization of latent space')
    plt.show()

if __name__ == "__main__":
    with torch.no_grad():
        recon_grid()
