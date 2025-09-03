import torch
from torch import nn
import collections
from typing import Iterable, List
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch.distributions import Normal
from sklearn.mixture import GaussianMixture
import numpy as np
import math
from tqdm import tqdm
from torch import nn, einsum
from einops import rearrange, repeat
from guided_diffusion.misc import exists, default
from guided_diffusion.modules import zero_module
from guided_diffusion.layers import FeedForward


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x

class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    :param use_relu: Whether to have `ReLU` layers or not
    :param bias: Whether to learn bias in linear layers or not

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        #use_batch_norm: bool = False,
        use_relu: bool = True,
        #use_relu: bool = False,
        bias: bool = True,
        RNA_mode = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.LayerNorm(n_out, eps=0.0001),
                            nn.LeakyReLU() if RNA_mode else None,
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.0001)
                            if use_batch_norm
                            else None,
                            nn.LeakyReLU() if use_relu else nn.ReLU(),
                            nn.Dropout(p=dropout) if dropout > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                    zip(layers_dim[:-1], layers_dim[1:])
                )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x
    
class MLP_VAE_encoder(nn.Module):
    """
    The MLP encoder layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(MLP_VAE_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.emb_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(0.1),
                ) for _ in range(layers)])
        self.ff= nn.Linear(self.hidden_dim, self.output_dim)
        self.ff_mean= nn.Linear(self.output_dim, self.output_dim)
        self.ff_std = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x):
        h = self.emb_layer(x)
        for layer in self.encoder:
            h = layer(h)
        h = self.ff(h)
        z_mean = self.ff_mean(h)
        z_std = self.ff_std(h)
        return h, z_mean, z_std
        # return z_mean, z_std


class MLP_AE_encoder(nn.Module):
    """
    The MLP encoder layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(MLP_AE_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.emb_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(0.1),
                ) for _ in range(layers)])
        self.ff= nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        h = self.emb_layer(x)
        for layer in self.encoder:
            h = layer(h)
        z = self.ff(h)

        return z

class MLP_VAE_decoder(nn.Module):
    """
    The MLP decoder layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(MLP_VAE_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.ff_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(0.1),
                ) for _ in range(layers)])
        self.ff_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, z):
        h = self.ff_in(z)
        for layer in self.decoder:
            h = layer(h)
        h = self.ff_out(h)
        return h

class Cell_type_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_cell_type, layers):
        super(Cell_type_Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


        self.emb_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder = nn.ModuleList([nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(0.1),
                ) for _ in range(layers)])
        self.ff= nn.Linear(self.hidden_dim, num_cell_type)

    def forward(self, x):
        h = self.emb_layer(x)
        for layer in self.encoder:
            h = layer(h)
        pred = self.ff(h)
        return pred

class CrossAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

    def forward(self, x, context=None):
        if context == None:
            context = x # self-attention
        x = self.attn(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x

# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout: Dropout rate to apply to each of the hidden layers
    :param distribution: Distribution of z
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout: float = 0.1,
        distribution: str = "normal",
    ):
        super().__init__()

        self.distribution = distribution
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent
    
class Encoder_selfattention(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_att_layers: int=3,
        n_hidden: int = 128,
        dropout: float = 0.1,
        n_heads: int = 8,
        d_head: int = 64,
        distribution: str = "normal",
        vae_mode = False
    ):
        super().__init__()
        self.vae_mode = vae_mode
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout =dropout,
            RNA_mode=False,
        )

        inner_dim = d_head * n_heads

        self.transformer_layers = nn.ModuleList([BasicTransformerBlock(n_hidden, n_heads, d_head=d_head, dropout=dropout) for _ in range(n_att_layers)])
        self.q_encoder = zero_module(nn.Linear(n_hidden, n_output))
        if self.vae_mode:
            self.mean_encoder = nn.Linear(n_output, n_output)
            self.var_encoder = nn.Linear(n_output, n_output)

            if distribution == "ln":
                self.z_transformation = nn.Softmax(dim=-1)
            else:
                self.z_transformation = identity

    def forward(self, x: torch.Tensor, cond_emb=None, *cat_list: int, ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q = q.unsqueeze(1)
        for layer in self.transformer_layers:
            q = layer(q,cond_emb)
        q = q.squeeze(1)
        q_a = self.q_encoder(q)
        if self.vae_mode:
            q_m = self.mean_encoder(q_a)
            q_v = self.var_encoder(q_a)
            # latent = self.z_transformation(reparameterize_gaussian(q_m, torch.exp(q_v) + 1e-4))
            eps = torch.randn(q_m.size(), requires_grad=False, device=q_m.device)
            sigma = torch.exp(q_v * 0.5)
            latent = q_m + sigma * eps

            return q_a, q_m, q_v, latent
        else:
            return q_a

class Encoder_selfattention_gmm(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_att_layers: int=3,
        n_hidden: int = 128,
        dropout: float = 0.1,
        n_heads: int = 8,
        d_head: int = 64,
        n_centroids = 30,
        distribution: str = "normal",
        vae_mode = False,
        gmm_mode = False
    ):
        super().__init__()
        self.vae_mode = vae_mode
        self.gmm_mode = gmm_mode
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            RNA_mode=False,
        )

        inner_dim = d_head * n_heads

        self.transformer_layers = nn.ModuleList([BasicTransformerBlock(n_hidden, n_heads, d_head=d_head, dropout=dropout) for _ in range(n_att_layers)])
        # self.self_attn = self_attn
        # self.q_encoder = zero_module(nn.Linear(n_hidden, n_output))
        self.q_encoder = nn.Linear(n_hidden, n_output)
        if self.vae_mode:
            self.mean_encoder = nn.Linear(n_output, n_output)
            self.var_encoder = nn.Linear(n_output, n_output)

            if distribution == "ln":
                self.z_transformation = nn.Softmax(dim=-1)
            else:
                self.z_transformation = identity

            self.n_centroids = n_centroids
            self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
            self.mu_c = nn.Parameter(torch.zeros(n_output, n_centroids)) # mu
            self.var_c = nn.Parameter(torch.ones(n_output, n_centroids)) # sigma^2
        
        self.cluster_decoder = FCLayers(
                n_in=n_centroids,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout=0,
            )

    def forward(self, x: torch.Tensor, cond_emb=None, return_loss=True,*cat_list: int, ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q = q.unsqueeze(1)
        for layer in self.transformer_layers:
            q = layer(q,cond_emb)
        q = q.squeeze(1)
        q_a = self.q_encoder(q)
        if self.vae_mode:
            q_m = self.mean_encoder(q_a)
            q_v = self.var_encoder(q_a) #log_var

            eps = torch.randn(q_m.size(), requires_grad=False, device=q_m.device)
            # sigma = torch.exp(q_v * 0.5)
            std = q_v.mul(0.5).exp_()
            # z = q_m + sigma * eps
            z = q_m.addcmul(std, eps)
            gamma, mu_c, var_c, pi = self.get_gamma(z)
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            if return_loss:
                kld_loss = self.kld_loss(gamma,mu_c,var_c,pi,q_m,q_v,gmm_mode=self.gmm_mode)
                return q_a, q_m, q_v, z, kld_loss, cluster_temp
            else:
                return q_a, q_m, q_v, z
        else:
            return q_a

    def kld_loss(self, gamma,mu_c,var_c,pi,mu,logvar,gmm_mode=True):
        if gmm_mode:
            var_c += 1e-8
            n_centroids = pi.size(1) 
            mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
            logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)
            # log p(z|c)
            logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                                torch.log(var_c) + \
                                                torch.exp(logvar_expand)/var_c + \
                                                (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
            
            # log p(c)
            logpc = torch.sum(gamma*torch.log(pi), 1)

            # log q(z|x) or q entropy    
            qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

            # log q(c|x)
            logqcx = torch.sum(gamma*torch.log(gamma), 1)
            
            # # log p(z|c)
            # logpzc = -0.5*torch.mean(gamma*torch.sum(math.log(2*math.pi) + \
            #                                     torch.log(var_c) + \
            #                                     torch.exp(logvar_expand)/var_c + \
            #                                     (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
            
            # # log p(c)
            # logpc = torch.mean(gamma*torch.log(pi), 1)

            # # log q(z|x) or q entropy    
            # qentropy = -0.5*torch.mean(1+logvar+math.log(2*math.pi), 1)

            # # log q(c|x)
            # logqcx = torch.mean(gamma*torch.log(gamma), 1)

            kld = -logpzc - logpc + qentropy + logqcx
        else:
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kld
    # def kld_loss(self, gamma,mu_c,var_c,pi,mu,logvar):
    #     posterior = torch.exp(-0.5 * torch.sum((mu.unsqueeze(1) - gmm.means_)**2 / gmm.covariances_, dim=2))
    #     posterior = posterior / torch.sum(posterior, dim=1, keepdim=True)

    #     # Compute KL divergence
    #     kl_loss = torch.sum(posterior * (torch.log(gmm.weights_) - torch.log(posterior + 1e-10)), dim=1)
    #     # kl_loss = torch.mean(kl_loss)

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1) # NxK
#         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) + 1e-8 # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init SCALE model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        z = self.encodeBatch(dataloader, device)
        gmm.fit(z)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
        print('Initial GMM')

    def encodeBatch(self, dataloader, device='cpu'):
        """
        Encode all data in dataloader
        """
        self.eval()
        z_all = []
        steps = 2
        for _,cond,_ in tqdm(dataloader):
            # x,_ = next(dataloader)
            x = cond['x_raw'].to(device)
            _, _, _, z = self.forward(x.float(),return_loss=False)
            z_all.append(z.data.cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)
        return z_all

class self_attention(nn.Module):
    def __init__(
        self,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads

        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(n_hidden, eps=0.0001)

    def forward(self,x,y):
        Q = self.w_q(x).view(x.shape[0],self.n_heads, x.shape[-1]//self.n_heads,-1)
        K = self.w_k(y).view(y.shape[0],self.n_heads, y.shape[-1]//self.n_heads,-1)
        V = self.w_v(y).view(y.shape[0],self.n_heads, y.shape[-1]//self.n_heads,-1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = self.layernorm(x+self.do(torch.matmul(attention, V).view(x.shape[0], x.shape[1])))

        return q_a

class Encoder_selfattention_gmm_1(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_att_layers: int=3,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 4,
        self_attn=True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout_rate,
            RNA_mode=False,
        )
        self.px_encoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )

        self.att_layer = nn.ModuleList([self_attention(n_hidden, dropout_rate, n_heads) for _ in range(n_att_layers)])
        self.do = nn.Dropout(dropout_rate)

        self.q_encoder = nn.Linear(n_hidden, n_output)

        self.mean_encoder = nn.Linear(n_output, n_output)
        self.var_encoder = nn.Linear(n_output, n_output)

        self.self_attn = self_attn
        n_centroids = 30 # 30
        self.n_centroids = n_centroids
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(n_output, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(n_output, n_centroids)) # sigma^2

    def forward(self, x: torch.Tensor, cond_emb=None, return_loss=True,*cat_list: int, ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        assert q.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        if self.self_attn or cond_emb==None:
            cond_emb = q
        for layer in self.att_layer:
            q = layer(q,cond_emb)

        q_a = self.q_encoder(q)
        q_m = self.mean_encoder(q_a)
        q_v = self.var_encoder(q_a)

        eps = torch.randn(q_m.size(), requires_grad=False, device=q_m.device)
        sigma = torch.exp(q_v * 0.5)
        z = q_m + sigma * eps
        gamma, mu_c, var_c, pi = self.get_gamma(z)
        if return_loss:
            kld_loss = self.kld_loss(gamma,mu_c,var_c,pi,q_m,sigma)
            return q_a, q_m, sigma, z, kld_loss
        else:
            return q_a, q_m, sigma, z

    
    def kld_loss(self, gamma,mu_c,var_c,pi,mu,logvar):
        
        var_c += 1e-8
        n_centroids = pi.size(1) 
        mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
        logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)
        # log p(z|c)
        logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                            torch.log(var_c) + \
                                            torch.exp(logvar_expand)/var_c + \
                                            (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
        
        # log p(c)
        logpc = torch.sum(gamma*torch.log(pi), 1)

        # log q(z|x) or q entropy    
        qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

        # log q(c|x)
        logqcx = torch.sum(gamma*torch.log(gamma), 1)

        kld = -logpzc - logpc + qentropy + logqcx
        return torch.mean(kld)
    
    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1) # NxK
#         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) + 1e-8 # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init SCALE model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        z = self.encodeBatch(dataloader, device)
        gmm.fit(z)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
        print('Initial GMM')

    def encodeBatch(self, dataloader, device='cpu'):
        """
        Encode all data in dataloader
        """
        self.eval()
        z_all = []
        for _,cond,_ in tqdm(dataloader):
            # x,_ = next(dataloader)
            x = cond['x_raw'].to(device)
            q_a, q_m, sigma, z = self.forward(x,return_loss=False)
            z_all.append(z.data.cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)
        return z_all
        
class Encoder_selfattention_zinb(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_att_layers: int=3,
        n_hidden: int = 128,
        dropout: float = 0.1,
        n_heads: int = 4,
        self_attn=True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            RNA_mode=False,
        )
        self.px_encoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )

        self.att_layer = nn.ModuleList([self_attention(n_hidden, dropout, n_heads) for _ in range(n_att_layers)])
        self.do = nn.Dropout(dropout)

        self.q_encoder = nn.Linear(n_hidden, n_output)

        self.mean_encoder = nn.Linear(n_output, n_output)
        self.var_encoder = nn.Linear(n_output, n_output)

        self.self_attn = self_attn
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))

        self.fc = nn.Sequential(nn.Linear(n_output, n_hidden),nn.GELU())
        self.mean_decoder = nn.Linear(n_hidden, n_input)
        self.dropout_decoder = nn.Linear(n_hidden, n_input)
        self.distribution = 'nb'
    
    def decode(self,z):
        h = self.fc(z)
        mu = self.mean_decoder(h)
        dropout_logits = self.dropout_decoder(h)
        return mu, dropout_logits
    
    def encode(self,x: torch.Tensor, cond_emb=None, *cat_list: int, ):
        q = self.encoder(x, *cat_list)
        assert q.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        if self.self_attn or cond_emb==None:
            cond_emb = q
        for layer in self.att_layer:
            q = layer(q,cond_emb)

        q_a = self.q_encoder(q)
        q_m = self.mean_encoder(q_a)
        q_v = self.var_encoder(q_a)

        return q_a, q_m, q_v

    def forward(self, x: torch.Tensor, cond_emb=None, *cat_list: int, ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q_a, q_m, q_v = self.encode(x,cond_emb,*cat_list)
        eps = torch.randn(q_m.size(), requires_grad=False, device=q_m.device)
        sigma = torch.exp(q_v * 0.5)
        z = q_m + sigma * eps
        de_mean, de_dropout = self.decode(z)

        return q_a, de_mean, de_dropout, q_m, q_v
    
    def get_latent_representation(self, x):
        _, mu, logvar = self.encode(x)
        return mu+logvar
    
    def reconstruction_loss(self, x, mu, dropout_logits):
        '''
        x: input data
        mu: output of decoder
        dropout_logits: dropout logits of zinb distribution
        '''
        theta = self.log_theta.exp()
        
        nb_logits = (mu+1e-5).log() - (theta+1e-5).log()
        
        if self.distribution == 'zinb':
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'nb':
            distribution = NegativeBinomial(total_count=theta, logits=nb_logits,validate_args=False)
        return distribution.log_prob(x).sum(-1).mean()
    
    def loss_function(self, x, mu, dropout_logits, mu_, logvar_):
        reconstruction_loss = self.reconstruction_loss(x, mu, dropout_logits)
        # kl_div = self.kl_d(mu_, logvar_)
        # return -reconstruction_loss + kl_div
        return -reconstruction_loss

class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout
    
# Multi-Dncoder-nb-selfattention
class Multi_Decoder_nb_SelfAttention(nn.Module):
    def __init__(
        self,
        n_input: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads


        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )

        self.atac_scale_decoder = nn.Sequential(
            nn.Linear( n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output), nn.Sigmoid()
        )

        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Softmax(dim=-1)
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden,1)
        )
        self.libaray_atac_scale_decoder =  nn.Sequential(
            nn.Linear(n_hidden,1)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor, *cat_list: int, gamma = None, libary_atac = None):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            #test version 210302

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        assert p_atac.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(p_atac).view(p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1)
        K = self.w_k(p_atac).view(p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1)
        V = self.w_v(p_atac).view(p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        p_atac = torch.matmul(attention, V).view(p_atac.shape[0], p_atac.shape[1])

        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))
        else:
            p_atac_scale = self.atac_scale_decoder(torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1))

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_scale = p_atac_scale*self.px_atac_decoder_aux(z)# for zinp and zip loss

        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_scale
        else:
            p_atac_mean = torch.exp(libaray_atac) * p_atac_scale

        return p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout