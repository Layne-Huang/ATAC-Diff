import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

import time
import math
import numpy as np
from tqdm import tqdm, trange
from itertools import repeat
from sklearn.mixture import GaussianMixture

from guided_diffusion.nn_new import MLP_VAE_encoder, MLP_VAE_decoder,Encoder_selfattention,Encoder_selfattention_gmm
from guided_diffusion.losses import elbo
import os


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

    def next(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

class Latent_VAE(nn.Module):
    def __init__(self, dims, data_path, layers=6, bn=False, dropout=0, binary=True):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(Latent_VAE, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.binary = binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None

        # self.encoder = MLP_VAE_encoder(x_dim, encode_dim, z_dim, layers)
        self.encoder = Encoder_selfattention(x_dim, z_dim, n_hidden=z_dim, n_att_layers=layers,vae_mode=True)
        # self.decoder = MLP_VAE_decoder(z_dim, decode_dim, x_dim, layers)
        self.decoder = Encoder_selfattention(z_dim, x_dim, n_hidden=z_dim, n_att_layers=layers)

        self.norm = nn.LayerNorm(x_dim)
        self.act = nn.SiLU()

        self.z_dim = z_dim
        print(data_path)
        self.data = torch.tensor(np.load(data_path,allow_pickle=True)['x0_embs'], dtype=torch.float32)
        self.cond_mean = self.data.mean(0)
        self.cond_std = self.data.std(0)
        print(self.cond_mean[:10])

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        _, mu, logvar = self.encoder(x)
        eps = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        sigma = torch.exp(logvar * 0.5)
        z = mu + sigma * eps
        recon_x = self.decoder(z)
        # recon_x = self.norm(recon_x)
        # recon_x = self.act(recon_x)

        return recon_x

    def loss_function(self, x):
        _, mu, logvar,_ = self.encoder(x)
        eps = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        sigma = torch.exp(logvar * 0.5)
        z = mu + sigma * eps
        recon_x = self.decoder(z)
        test_x = recon_x[0]
        val_x = recon_x[:,0]
        # recon_x = self.norm(recon_x)
        # recon_x = self.act(recon_x)
        likelihood, kl_loss = elbo(recon_x, x, (mu, logvar))

        return (likelihood, kl_loss)

    def get_x_emb(self,b,device):
        z = torch.randn(b, self.z_dim, device=device)
        x_emb = self.decoder(z)
        # x_emb = self.norm(x_emb)
        # x_emb = self.act(x_emb)
        # x_emb = x_emb * self.cond_std.to(device) + self.cond_mean.to(device)
        return x_emb
        
    def predict(self, dataloader, device='cpu', method='kmeans'):
        """
        Predict assignments applying k-means on latent feature

        Input: 
            x, data matrix
        Return:
            predicted cluster assignments
        """

        if method == 'kmeans':
            from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
            feature = self.encodeBatch(dataloader, device)
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
            pred = kmeans.fit_predict(feature)
        elif method == 'gmm':
            logits = self.encodeBatch(dataloader, device, out='logit')
            pred = np.argmax(logits, axis=1)

        return pred

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, dataloader,
            lr=0.002, 
            weight_decay=5e-4,
            device='cpu',
            beta = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            patience=500,
            outdir=None,
       ):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) 
        Beta = DeterministicWarmup(n=n, t_max=beta)
        
        iteration = 0
        n_epoch = int(np.ceil(max_iter/len(dataloader)))
        n_epoch = max_iter
        # early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
#                 epoch_loss = 0
                epoch_recon_loss, epoch_kl_loss = 0, 0
                epoch_recon_loss_list= []
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                for i, x in tk0:
#                     epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    x = x.float().to(device)
                    # x= (x - self.cond_mean.to(
                    #     device)) / self.cond_std.to(device)
                    # print(x)
                    optimizer.zero_grad()
                    
                    recon_loss, kl_loss = self.loss_function(x)

                    loss = recon_loss + kl_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10) # clip
                    optimizer.step()
                    
                    epoch_kl_loss += kl_loss.item()
                    epoch_recon_loss += recon_loss.item()

                    epoch_recon_loss_list.append(recon_loss.item())

                    tk0.set_postfix_str('loss={:.5f} recon_loss={:.5f} kl_loss={:.3f}'.format(
                            loss, recon_loss, kl_loss))
                    tk0.update(1)
                    
                    iteration+=1
                tq.set_postfix_str('recon_loss {:.5f} kl_loss {:.5f}'.format(
                    epoch_recon_loss/((i+1)), epoch_kl_loss/((i+1))))
                tq.update(1)
 
        states = [
        self.state_dict(),
        optimizer.state_dict(),
        epoch]

            
        print("Saving the last checkpoint of epoch {}".format(epoch))
        torch.save(states, os.path.join(outdir, "Latent_VAE_x0_mu_128.pt"))

                # wandb.log({"loss": loss}, commit=True)
                # wandb.log({"recon_loss": recon_loss/len(x)}, commit=True) 
                # wandb.log({"kl_loss": kl_loss/len(x)}, commit=True)
                # if early_stopping(epoch_recon_loss/((i+1)*len(x)), self, epoch):
                #     print('Early stop at epoch ', str(epoch))
                #     return epoch


    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for x in dataloader:
            x = x.view(x.size(0), -1).float().to(device)
            z, mu, logvar = self.encoder(x)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)

        output = torch.cat(output).numpy()

        return output