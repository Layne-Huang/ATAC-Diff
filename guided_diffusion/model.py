import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .nn import (
    linear,
    timestep_embedding,
)

from .nn_new import (
    Encoder_selfattention,
    Encoder_selfattention_gmm,
    Encoder_selfattention_gmm_1
)

class Model(nn.Module):
    def __init__(self, input_dim, raw_dim, hidden_dims, encode_depth = 6, side_depth = 1, n_centroids=8, cond_dim=2, cond_tokens=2):
        super().__init__()

        x0_dim = hidden_dims[0]
        time_dim = hidden_dims[0]
        num_con = cond_tokens # the number of conditional information, e.g. cell type
        raw_dim = input_dim if raw_dim==0 else raw_dim

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dims[0], 4 * hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(4 * hidden_dims[0], time_dim),
        )


        self.atac_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        self.con_encoder = nn.Sequential(
                    nn.Linear(cond_dim, hidden_dims[0] * cond_tokens),
                    Rearrange('b (n d) -> b n d', n=cond_tokens, d=hidden_dims[0]),
                )
        
        self.encoder = Encoder_selfattention(hidden_dims[0], hidden_dims[0], n_hidden=hidden_dims[0], n_att_layers=encode_depth)
        # self.side_encoder = Encoder_selfattention(input_dim, hidden_dims[0], n_hidden=hidden_dims[0], n_att_layers=side_depth)
        # self.side_encoder = Encoder_selfattention_gmm(raw_dim, hidden_dims[-1], n_att_layers=side_depth, vae_mode=True,gmm_mode=False, n_centroids=n_centroids)
        self.side_encoder =  Encoder_selfattention_gmm_1(raw_dim, hidden_dims[-1])
        self.z_ff = nn.Linear(hidden_dims[-1], hidden_dims[0])

        self.side_decoder = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Linear(hidden_dims[1], raw_dim),
            # nn.Sigmoid(),
            )

        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dims[0], hidden_dims[1]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[1], input_dim),
        #     )

        # self.side_decoder = nn.Sequential(
        #     nn.Linear(hidden_dims[0], hidden_dims[0]),

        #     )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[0], input_dim),
            )
        
        self.atac_scale_decoder = nn.Sequential(nn.Linear(hidden_dims[0],1))
        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]), nn.Linear(hidden_dims[0], raw_dim)
        )
        self.hidden_dims = hidden_dims
        
    def forward(self, xt, x0, t, x_raw=None, cell_type=None, batch=None, latent=[False]):
        temb = self.time_embed(timestep_embedding(t, self.hidden_dims[0]).squeeze(1))

        h = self.atac_encoder(xt.float())
        h = h+temb

        if latent[0]==True:
            x0_emb_align = h
            x0_emb = h
            x0_h, x0_mu, x0_log_var,kld_loss = None, None, None, None
        elif x_raw is not None and x_raw.dim()==2:
            # x0_h, x0_mu, x0_log_var,x0_emb,kld_loss,cluster_temp = self.side_encoder(x_raw.float())
            x0_h, x0_mu, x0_log_var,x0_emb,kld_loss = self.side_encoder(x_raw.float())
            # x0_emb = F.normalize(x0_emb, p=2, dim=1)
            x0_emb_align = self.z_ff(x0_emb)
            # x0_emb_norm = F.normalize(x0_emb_align, p=2, dim=1)

        elif x0==None:
            x0 = torch.zeros((xt.size(0),11285)).to(h.device)
            x0_h, x0_mu, x0_log_var,x0_emb,kld_loss,cluster_temp = self.side_encoder(x0.float())
            # x0_emb = F.normalize(x0_emb, p=2, dim=1)
            x0_emb_align = self.z_ff(x0_emb)
            # x0_emb_norm = F.normalize(x0_emb_align, p=2, dim=1)

        elif x0.size(1)>10000:
            x0_h, x0_mu, x0_log_var,x0_emb,kld_loss = self.side_encoder(x0.float())
            # x0_emb = F.normalize(x0_emb, p=2, dim=1)
            x0_emb_align = self.z_ff(x0_emb)
            # x0_emb_norm = F.normalize(x0_emb_align, p=2, dim=1)

        else: # for sampling
            x0_emb = x0
            x0_h, x0_mu, x0_log_var = None, None, None
            # x0_emb = F.normalize(x0_emb, p=2, dim=1)
            x0_emb_align = self.z_ff(x0_emb)
            # x0_emb_norm = F.normalize(x0_emb_align, p=2, dim=1)
        
       
        # c = None
        # if c is not None:
        #     c_emb = self.con_encoder(c.unsqueeze(1)).squeeze(1)
        #     h = h+c_emb+x0_emb
        # else:
        #     h = h

        # latent diff or not

        if cell_type is not None and batch is not None:
            cond = torch.stack((cell_type,batch),dim=1).float()
            cond_emb = self.con_encoder(cond)
            h = self.encoder(h,cond_emb=torch.cat([cond_emb,x0_emb_align.unsqueeze(1)],dim=1))
            # h = self.encoder(h,cond_emb=x0_emb.unsqueeze(1))
        else:
            h = self.encoder(h,cond_emb=x0_emb_align.unsqueeze(1))
            # h = self.encoder(h)
        # h = self.encoder(h,cond_emb=x0_emb.unsqueeze(1))
        # h = self.encoder(h)
        output = self.decoder(h)

        
        # x0_recon = None
        # return output
        if self.training:
            if latent[0]==True:
                x0_recon=None
            else:
                x0_recon = self.side_decoder(x0_emb_align)

            # p_atac = self.side_decoder(x0_emb)
            # p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))
            # x0_recon = p_atac_scale*self.px_atac_decoder_aux(x0_emb)# for zinp and zip loss

            return output, x0_emb, x0_mu, x0_log_var,x0_h, x0_recon, kld_loss
        else:
            if latent[0]==True:
                x0_recon=None
                return output,x0_recon
            x0_recon = self.side_decoder(x0_emb_align)
            return output,x0_recon

    def get_emb(self, x0):
        # x0_h, x0_mu, x0_log_var,x0_emb = self.side_encoder(x0.float(),return_loss=False)
        x0_h, x0_mu, x0_log_var,x0_emb = self.side_encoder(x0.float(),return_loss=False)
        # x0_emb_align = self.z_ff(x0_emb)
        # x0_emb_norm = F.normalize(x0_emb_align, p=2, dim=1)
        # x0_emb = x0_mu
        # x0_emb = F.normalize(x0_emb, p=2, dim=1)
        # x0_emb = self.side_encoder(x0.float())

        return F.normalize(x0_mu, p=2, dim=1)

    def recover(self,x0):
        x0_h, x0_mu, x0_log_var,x0_emb,kld_loss,cluster_temp = self.side_encoder(x0.float())
        x0_recon = self.side_decoder(x0_emb)
        return x0_recon


    # def sample(self, xt, x0, c, t):
    #     if t == None:
    #         x0_h, x0_mu, x0_log_var,x0_emb = self.side_emb(x0,return_loss=False)
    #         x_recon = self.side_decoder(x0_emb)
    #         return x_recon

    #     temb = self.time_embed(timestep_embedding(t, self.hidden_num[0]).squeeze(1))
    #     h = self.atac_encoder(xt)
    #     # temb = temb.expand_as(h)
    #     if x0==None:
    #         x0_emb = torch.rand_like(h).to(h.device)
    #     elif x0.size()==xt.size():
    #         # x0_h, x0_mu, x0_log_var,x0_emb = self.side_emb(x0,return_loss=False)
    #         x0_emb = self.side_encoder(x0.float())

    #     else:
    #         x0_emb = x0

    #     if c is not None:
    #         c_emb = self.con_encoder(c.unsqueeze(1)).squeeze(1)
    #         h = h+c_emb+x0_emb
    #     else:
    #         h = h+x0_emb

    #     h = self.encoder(h)
    #     h = self.decoder(h)
        
    #     return h,x0_emb, x_recon