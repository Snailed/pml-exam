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
from tqdm import tqdm, trange

parser = argparse.ArgumentParser(description='VAE MNIST Lightning')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--checkpoint-path', type=str, default='',
                    help='resume from checkpoint (default: None)')
args = parser.parse_args()


# def C(lam):
#     no_twos = (torch.logical_and(lam != 0.5, torch.logical_and(lam != 0.0, lam != 1.0))).float()
#     #print(f"no twos {no_twos}")
#     twos = (lam == 0.5).float()
#     first_part = 2*torch.arctanh(1 - 2 * (no_twos * lam))/(1 - 2*(no_twos * lam))
#     second_part = twos * 2
#     #print(f"Recon_x: {no_twos * torch.nan_to_num(first_part, nan=0.0, posinf=0.0, neginf=0.0) + second_part}")
#     return no_twos * torch.nan_to_num(first_part, nan=0.0, posinf=0.0, neginf=0.0) + second_part + (lam == 1.0).float()

# def log_C(lam):
#     mask = (torch.logical_and(lam != 0.0, lam != 1.0)).float()
#     return torch.nan_to_num(torch.log(C(lam) * mask))


# def log_C(lam):
#     return (2 * torch.arctanh(1 - 2 * lam))/(1 - 2 * lam)

def C( x ):
    return (2. * torch.arctanh(1. - 2.*x))/(1. - 2.*x) 
def logC( x ):
    # Numerically stable implementation from https://github.com/Robert-Aduviri/Continuous-Bernoulli-VAE/blob/master/notebooks/Continuous_Bernoulli.ipynb
    if abs(x - 0.5) < 1e-3:
        # Taylor Approximation around 0.5
        value = torch.log(torch.tensor(2.))
        taylor = torch.tensor(1); nu = 1. - 2. * x; e = 1; k = nu**2
        for i in range(1, 10): 
            e *= k; taylor += e / (2. * i + 1) 
        return value + torch.log( taylor )
    return torch.log( C(x) )

# Also from https://github.com/Robert-Aduviri/Continuous-Bernoulli-VAE/blob/master/notebooks/Continuous_Bernoulli.ipynb
def sumlogC( x , eps = 1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation
        
    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    x = torch.clamp(x, eps, 1.-eps) 
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    return far_values.sum() + close_values.sum()

class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 4))
        self.decoder = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 28 * 28), nn.Sigmoid())
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x.view(x.size(0), -1))
        return embedding

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        #BCE_p = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # BCE = -1 * torch.sum(x.view(-1,784) * torch.log(recon_x) + (1-x.view(-1,784))*torch.log(1 - recon_x))
        # BCE = -1 * torch.sum(x.view(-1,784) * torch.log(recon_x) + (1-x.view(-1,784))*torch.log(1 - recon_x))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # return BCE + KLD + sumlogC(recon_x)
        #return KLD + F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') #+ torch.sum(torch.distributions.ContinuousBernoulli(probs=recon_x)._cont_bern_log_norm())
        return KLD - torch.sum(torch.distributions.ContinuousBernoulli(probs=recon_x).log_prob(x))
        #return KLD + torch.sum(torch.distributions.Beta(x.view(-1, 784), x.view(-1, 784)).log_prob(recon_x))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x_enc = self.encoder(x)
        mu, logvar = x_enc[:,:2], x_enc[:,2:4]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        loss = self.loss_function(recon_x, x, mu, logvar)
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
        format = "{:.3f}".format
        axs[0, i].set(yticklabels=[format(y) for y in y_samples], xticks=np.arange(20)*30 + 14, yticks=np.arange(20)*30 + 16)
        axs[0,i].set_xticklabels([format(x) for x in x_samples], rotation=90)
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
    sample = model.decoder(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
    grid = make_grid(torch.distributions.ContinuousBernoulli(sample.view(20 * 20, 1, 28, 28)).mean, nrow=20)
    show(grid, x_uniform.numpy(), x_normal.numpy(), y_normal.numpy())
    plt.title('Continuous Bernoulli Likelihood')
    plt.show()


def monte_carlo_sampling(test_set, n_samples=100):
    for x, y in DataLoader(test_set, batch_size=5000):
        # samples = torch.distributions.Normal(0,1).sample((n_samples, 2))
        # probs = []
        # means = []
        # vars = []
        # for image in tqdm(x[:500]):
        #     x_recon = model.decoder(samples)
        #     probs.append(torch.sum(torch.distributions.ContinuousBernoulli(probs=x_recon).log_prob(image.view(-1, 784)), dim=1))
        #     means.append(np.mean(np.array(probs)))
        #     vars.append(np.var(np.array(probs)))
        # plt.plot(means)
        # plt.show()
        samples = torch.distributions.Normal(0,1).sample((n_samples, 2))
        x_recon = model.decoder(samples)
        probs = []
        means = []
        vars = []
        for i in trange(n_samples):
            probs.append(torch.sum(torch.distributions.ContinuousBernoulli(probs=x_recon[i]).log_prob(x.view(-1, 784)), dim=1))
            means.append(np.mean(np.array(probs)))
            vars.append(np.var(np.array(probs)))
        print(f"Final mean {means[-1]}")
        print(f"Final variance {vars[-1]}")
        plt.plot(vars)
        plt.title('Estimator variance')
        plt.ylabel('$Var[\\log p(x|z)]$')
        plt.xlabel('n_samples of $z$')
        plt.show()
        plt.plot(means)
        plt.xlabel('n_samples of $z$')
        plt.ylabel('$E[\\log p(x|z)]$')
        plt.title('Estimator mean')
        plt.show()




if __name__ == "__main__":
    with torch.no_grad():
        monte_carlo_sampling(test, n_samples=400)
        # recon_grid()
        # for x, y in DataLoader(test, batch_size=5000):
        #     x = x.view(x.size(0), -1)
        #     x_enc = model.encoder(x)
        #     mu, logvar = x_enc[:,:2], x_enc[:,2:4]
        #     z = model.reparameterize(mu, logvar)
        #     recon_x = model.decoder(z)
        #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #     ELBO_bernoulli = -F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') - KLD
        #     ELBO_cont_bernoulli = torch.sum(torch.distributions.ContinuousBernoulli(probs=recon_x).log_prob(x.view(-1, 784))) - KLD
        #     print(f"KLD {KLD}")
        #     print(f"Cont likelihood {torch.distributions.ContinuousBernoulli(probs=recon_x).log_prob(x.view(-1, 784))}")
        #     print(f"ELBO (bernoulli) {ELBO_bernoulli}")
        #     print(f"ELBO (cont_bernoulli) {ELBO_cont_bernoulli}")

