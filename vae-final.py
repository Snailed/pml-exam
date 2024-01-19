from __future__ import print_function
import os
import os.path
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
from lightning.pytorch.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='VAE MNIST Lightning')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--checkpoint-path', type=str, default='',
                    help='resume from checkpoint (default: None)')
parser.add_argument('--model', choices=['bernoulli', 'continuous-bernoulli', 'beta-softplus', 'beta-sigmoid'], default='bernoulli',
                    help='Model to train (default: bernoulli)')
parser.add_argument('--recon', action=argparse.BooleanOptionalAction, default=True,
                    help='Produce reconstructions (default: True)')
parser.add_argument('--monte-carlo', action=argparse.BooleanOptionalAction, default=True,
                    help='Produce monte carlo estimation (default: True)')
parser.add_argument('--importance-sampling', action=argparse.BooleanOptionalAction, default=True,
                    help='Produce importance sampling estimation (default: True)')
parser.add_argument('--scatter', action=argparse.BooleanOptionalAction, default=True,
                    help='Produce latent space scatter plot (default: True)')
args = parser.parse_args()


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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class Bernoulli(VAE):
    def __init__(self):
        super().__init__()

    def to_binary(self, x):
        return (x >= 0.5).float()
    def reconstruct(self, z):
        return torch.distributions.Bernoulli(probs=self.decoder(z)).sample((1,))

    # def training_step(self, batch, batch_idx):
    #     # training_step defines the train loop. It is independent of forward
    #     x, y = batch
    #     x = x.view(x.size(0), -1)
    #     x_enc = self.encoder(x)
    #     mu, logvar = x_enc[:,:2], x_enc[:,2:4]
    #     z = self.reparameterize(mu, logvar)
    #     p = self.decoder(z)
    #     loss = self.loss_function(p, x, mu, logvar)

    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    def loss_function(self, recon_x, x, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD - torch.sum(torch.distributions.Bernoulli(probs=recon_x).log_prob(self.to_binary(x.view(-1, 784))))

class BCE(VAE):
    def __init__(self):
        super().__init__()

    def reconstruct(self, z):
        return self.decoder(z)

    def loss_function(self, recon_x, x, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD + F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

class ContinuousBernoulli(VAE):
    def __init__(self):
        super().__init__()

    def reconstruct(self, z):
        return torch.distributions.ContinuousBernoulli(probs=self.decoder(z)).sample((1,))

    def loss_function(self, recon_x, x, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD - torch.sum(torch.distributions.ContinuousBernoulli(probs=recon_x).log_prob(x.view(-1, 784)))

class BetaSoftplus(VAE):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 4))
        self.decoder_a = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 28 * 28), nn.Softplus())
        self.decoder_b = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 28 * 28), nn.Softplus())

    def reconstruct(self, z):
        return torch.distributions.Beta(self.decoder_a(z), self.decoder_b(z)).mean

    def loss_function(self, a, b, x, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        clamped_x = torch.clamp(x.view(-1, 784), min=1e-7, max=1-1e-7)
        return KLD - torch.sum(torch.distributions.Beta(a, b).log_prob(clamped_x))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x_enc = self.encoder(x)
        mu, logvar = x_enc[:,:2], x_enc[:,2:4]
        z = self.reparameterize(mu, logvar)
        a = self.decoder_a(z)
        b = self.decoder_b(z)
        loss = self.loss_function(a, b, x, mu, logvar)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

class BetaSigmoid(VAE):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 4))
        self.decoder_a = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 28 * 28), nn.Sigmoid())
        self.decoder_b = nn.Sequential(nn.Linear(2, 400), nn.ReLU(), nn.Linear(400,400), nn.ReLU(), nn.Linear(400, 28 * 28), nn.Sigmoid())

    def reconstruct(self, z):
        return torch.distributions.Beta(self.decoder_a(z), self.decoder_b(z)).mean

    def loss_function(self, a, b, x, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        clamped_x = torch.clamp(x.view(-1, 784), min=1e-7, max=1-1e-7)
        return KLD - torch.sum(torch.distributions.Beta(a, b).log_prob(clamped_x))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x_enc = self.encoder(x)
        mu, logvar = x_enc[:,:2], x_enc[:,2:4]
        z = self.reparameterize(mu, logvar)
        a = self.decoder_a(z)
        b = self.decoder_b(z)
        loss = self.loss_function(a, b, x, mu, logvar)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, test = random_split(dataset, [55000, 5000])

Model = Bernoulli
if args.model == 'bernoulli':
    Model = Bernoulli
if args.model == 'continuous-bernoulli':
    Model = ContinuousBernoulli
if args.model == 'beta-softplus':
    Model = BetaSoftplus
if args.model == 'beta-sigmoid':
    Model = BetaSigmoid

trainer = pl.Trainer(max_epochs=args.epochs)
if args.checkpoint_path == '':
    model = Model()
    trainer.fit(model, DataLoader(train, num_workers=7, batch_size=args.batch_size))
else:
    model = Model.load_from_checkpoint(args.checkpoint_path)
    model.eval()

def show_grid(imgs, x_samples, y_samples):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        format = "{:.2f}".format
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
    plt.title(args.model.capitalize())
    plt.xlabel('$z_0$')
    plt.ylabel('$z_1$')
    plt.legend()
    plt.savefig(f'lightning_logs/version_{trainer.logger.version}/scatter')
    plt.clf()

def recon_grid():
    k = 20
    x_uniform = torch.linspace(0.00001, 0.99999, k)
    y_uniform = torch.linspace(0.00001, 0.99999, k)
    x_normal = torch.distributions.normal.Normal(0, 1).icdf(x_uniform)
    y_normal = torch.distributions.normal.Normal(0, 1).icdf(y_uniform)
    grid_x, grid_y = torch.meshgrid(x_normal, y_normal, indexing='ij')
    # Bernoulli
    samples = torch.tensor([])
    if args.model == 'bernoulli':
        p = model.decoder(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
        samples = model.to_binary(torch.distributions.Bernoulli(p).sample((1,)))
    if args.model == 'continuous-bernoulli':
        lam = model.decoder(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
        samples = torch.distributions.ContinuousBernoulli(lam).sample((1,))
    if args.model == 'beta-softplus' or args.model == 'beta-sigmoid':
        model_a = model.decoder_a(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
        model_b = model.decoder_b(torch.dstack([grid_y, grid_x]).reshape([-1,2]))
        samples = torch.distributions.Beta(model_a, model_b).sample((1,))
    grid = make_grid(samples.view(20 * 20, 1, 28, 28), nrow=20)
    show_grid(grid, x_normal.numpy(), y_normal.numpy())
    plt.title(args.model.capitalize())
    plt.savefig(f'lightning_logs/version_{trainer.logger.version}/recon')
    plt.clf()

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
        a, b, lam, recon = ([], [], [], [])
        if args.model == 'beta-sigmoid' or args.model == 'beta-softplus':
            a = model.decoder_a(samples)
            b = model.decoder_b(samples)
        if args.model == 'continuous-bernoulli':
            lam = model.decoder(samples)
        else:
            recon = model.decoder(samples)
        probs = []
        means = []
        vars = []
        for i in trange(n_samples):
            if args.model == 'beta-sigmoid' or args.model == 'beta-softplus':
                probs.append(torch.sum(torch.distributions.Beta(a[i], b[i]).log_prob(torch.clamp(x.view(-1, 784), min=1e-7, max=1-1e-7)), dim=1))
            if args.model == 'continuous-bernoulli':
                probs.append(torch.sum(torch.distributions.ContinuousBernoulli(lam[i]).log_prob(x.view(-1, 784)), dim=1))
            if args.model == 'bernoulli': # Cannot use bernoulli since x needs to be in {0,1}
                probs.append(torch.sum(torch.distributions.Bernoulli(recon[i]).log_prob(model.to_binary(x.view(-1, 784))), dim=1))
            means.append(np.mean(np.array(probs)))
            vars.append(np.var(np.array(probs)))
        print(f"Final mean {means[-1]}")
        print(f"Final variance {vars[-1]}")
        plt.plot(vars)
        plt.title(f'Estimator variance ({args.model.capitalize()})')
        plt.ylabel('$Var[\\log p(x|z)]$')
        plt.xlabel('n_samples of $z$')
        plt.savefig(f'lightning_logs/version_{trainer.logger.version}/vars')
        plt.clf()
        plt.plot(means)
        plt.xlabel('n_samples of $z$')
        plt.ylabel('$E[\\log p(x|z)]$')
        plt.title(f'Estimator mean ({args.model.capitalize()})')
        plt.savefig(f'lightning_logs/version_{trainer.logger.version}/means')
        plt.clf()

def importance_sampling(test, n_samples=200):
    for x, y in DataLoader(test, batch_size=5000):
        # x_enc = model.encoder(x[:n_samples].view(-1, 784))
        # mus, logvars = x_enc[:,:2], x_enc[:, 2:4]
        # samples = model.reparameterize(mus, logvars)
        # importance_weights = (torch.distributions.Normal(mus, (0.5 * logvars).exp()).log_prob(samples) - torch.distributions.Normal(0,1).log_prob(samples)).sum(dim=1).exp()
        # a, b, lam, recon = ([], [], [], [])
        # if args.model == 'beta-sigmoid' or args.model == 'beta-softplus':
        #     a = model.decoder_a(samples)
        #     b = model.decoder_b(samples)
        # if args.model == 'continuous-bernoulli':
        #     lam = model.decoder(samples)
        # else:
        #     recon = model.decoder(samples)
        # probs = []
        # means = []
        # vars = []
        # for i in trange(5000):
        #     print(importance_weights[:i+1])
        #     if args.model == 'beta-sigmoid' or args.model == 'beta-softplus':
        #         probs.append(importance_weights[:i+1] * torch.sum(torch.distributions.Beta(a[i], b[i]).log_prob(torch.clamp(x.view(-1, 784)[:i+1], min=1e-7, max=1-1e-7)), dim=1))
        #     if args.model == 'continuous-bernoulli':
        #         probs.append(importance_weights[:i+1] * torch.sum(torch.distributions.ContinuousBernoulli(lam[i]).log_prob(x.view(-1, 784)[:i+1]), dim=1))
        #     if args.model == 'bernoulli':
        #         probs.append(importance_weights[:i+1] * torch.sum(torch.distributions.Bernoulli(recon[i]).log_prob(model.to_binary(x.view(-1, 784)[:i+1])), dim=1))
        #     means.append(np.mean(np.array(probs)))
        #     vars.append(np.var(np.array(probs)))
        # print(f"Final mean {means[-1]}")
        # print(f"Final variance {vars[-1]}")
        n_images = 1000
        # log_p = lambda *args: 0
        # if args.model == 'bernoulli':
        #     log_p = lambda x, par: torch.distributions.Bernoulli(par).log_prob(model.to_binary(x))
        # if args.model == 'continuous-bernoulli':
        #     log_p = lambda x, par: torch.distributions.ContinuousBernoulli(par).log_prob(model.to_binary(x))
        # if args.model == 'beta-sigmoid' or args.model == 'beta-softplus':
        #     log_p = lambda x, par1, par2: torch.distributions.Beta(par1, par2).log_prob(model.to_binary(x))
        means = []
        for n in trange(50):
            x = x[:n_images] # just consider one image for now
            n_samples = n * 20 + 1
            z_enc = model.encoder(x.view(-1,784))
            mu, logvar = z_enc[:, :2], z_enc[:, 2:4]
            z = torch.distributions.Normal(mu,(0.5*logvar).exp()).sample((n_samples,))
            log_p_part = 0
            if args.model == 'bernoulli' or args.model == 'continuous-bernoulli':
                if args.model == 'bernoulli':
                    log_p = lambda x, par: torch.distributions.Bernoulli(par).log_prob(model.to_binary(x))
                if args.model == 'continuous-bernoulli':
                    log_p = lambda x, par: torch.distributions.ContinuousBernoulli(par).log_prob(x)
                z_dec = model.decoder(z)
                log_p_part  = log_p(x.view(-1,784), z_dec.view(n_samples, n_images, 784))
            if args.model == 'beta-sigmoid' or args.model == 'beta-softplus':
                log_p = lambda x, par1, par2: torch.distributions.Beta(par1, par2).log_prob(torch.clamp(x, min=1e-5, max=1-1e-5))
                z_dec1 = model.decoder_a(z)
                z_dec2 = model.decoder_b(z)
                # print(z_dec1)
                log_p_part  = log_p(x.view(-1,784), z_dec1.view(n_samples, n_images, 784), z_dec2.view(n_samples, n_images, 784))
                # print(log_p_part)
            importance_weights = (torch.sum(torch.distributions.Normal(0,1).log_prob(z), dim=2)) - torch.sum(torch.distributions.Normal(mu,(0.5*logvar).exp()).log_prob(z), dim=2)
            #print(torch.mean(torch.mean(importance_weights.exp() * torch.sum(log_p(x.view(-1,784), z_dec.view(n_samples, n_images, 784)), dim=2), dim=0), dim=0))
            means.append(torch.mean(torch.mean(importance_weights.exp() * torch.sum(log_p_part, dim=2), dim=0), dim=0))
            
        #print(torch.sum(torch.sum(log_p(x.view(-1,784), z_dec), dim=1)))
        # plt.plot(vars)
        # plt.title(f'Estimator variance ({args.model.capitalize()})')
        # plt.ylabel('$Var[\\log p(x|z)]$')
        # plt.xlabel('n_samples of $z$')
        # plt.savefig(f'lightning_logs/version_{trainer.logger.version}/vars')
        # plt.clf()
        print(means[-1])
        plt.plot(np.linspace(0,50*20, 50), means)
        plt.xlabel('n_samples of $z$')
        plt.ylabel('$E[\\log p(x|z)]$')
        plt.title(f'Estimator mean ({args.model.capitalize()})')
        plt.savefig(f'means-{trainer.logger.version}-{args.model}')
        plt.clf()

if __name__ == "__main__":
    with torch.no_grad():
        if args.recon:
            recon_grid()
        if args.monte_carlo:
            monte_carlo_sampling(test, n_samples=200)
        if args.importance_sampling:
            importance_sampling(test, n_samples=200)
        if args.scatter:
            scatter_latent()
