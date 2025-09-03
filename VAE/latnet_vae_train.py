import scanpy as sc
import scanpy
import torch
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import numpy as np
from scipy.stats import spearmanr
sys.path.append("..")

from torch.utils.data import Dataset
# from guided_diffusion.losses import normal_kl, discretized_gaussian_log_likelihood,compute_mmd
from VAE import Latent_VAE
import torch.utils.tensorboard as tb

class X0_emb_Dataset(Dataset):
    def __init__(self, x0_emb):
        self.x0_emb = x0_emb
        # self.shape = adata.shape
        
    def __len__(self):
        return self.x0_emb.shape[0]
    
    def __getitem__(self, idx):
        x = self.x0_emb[idx].squeeze()
        return x

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Train the latent VAE')
    parser.add_argument('--data_dir', '-d', help='Load the data')
    parser.add_argument('-max_iter','-m', help='Maximum iterations', default=1000, type=int)
    parser.add_argument('-batch_size','-b', help='Batch size', default=128, type=int)
    parser.add_argument('-lr', help='Learning rate', default=1e-4, type=float)
    parser.add_argument('-gpu', help='GPU device', default=0, type=int)
    parser.add_argument('-outdir','-o', help='Output directory', default='output', type=str)

    
    return parser.parse_args()

def main():
    args = parse_args_and_config()
    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device='cpu'

    args.outdir = '/om/user/layne_h/project/atacdiff/checkpoint/forebrain'
    # args.data_dir = '/om/user/layne_h/project/atacdiff/output/forebrain/forebrain_emb_10k.npz'
    # args.data_dir = '/om/user/layne_h/project/atacdiff/output/forebrain/forebrain_sample_x0_x0.npz'
    args.data_dir = '/om/user/layne_h/project/atacdiff/output/forebrain/mu_128_x0emb.npz'
    
    # data = np.load(args.data_dir)['cell_emb']
    data = np.load(args.data_dir)['x0_embs']
    x0_emb_dataset = X0_emb_Dataset(data)
    print(data.shape)
    dataloader = torch.utils.data.DataLoader(x0_emb_dataset, batch_size=args.batch_size, shuffle=True)
    model = Latent_VAE(dims=[data.shape[1],256,256,64],data_path=args.data_dir, binary=False).to(device)

    if os.path.exists(os.path.join(args.outdir, "Latent_VAE_x0_mu_128.pt")):
        model.load_state_dict(torch.load(os.path.join(args.outdir, "Latent_VAE_x0_mu_128.pt"))[0])
    else:
        model.fit(dataloader, max_iter=args.max_iter, lr=args.lr, device=device, outdir=args.outdir)

    # model.load_state_dict(torch.load(os.path.join(args.outdir,'Latent_VAE.pt'))[0])

    samples = data.shape[0]
    # samples = 256
    recon_list = []
    model.eval()
    with torch.no_grad():
        for n in range(samples//args.batch_size):
            x_recon = model.get_x_emb(args.batch_size,device)
            recon_list.append(x_recon.cpu().detach())
        left = samples%args.batch_size
        if left>0:
            x_recon = model.get_x_emb(left,device)
            recon_list.append(x_recon.cpu().detach())

    recon_list = torch.cat(recon_list).numpy()
    real_list = data[:samples]
    # print(real_list.mean(axis=0))
    # print(recon_list.mean(axis=0))
    print(real_list[0,:20])
    print(recon_list[0,:20])
    print('spearman=',spearmanr(real_list.mean(axis=0),recon_list.mean(axis=0)).correlation)
    print('pearson=',np.corrcoef(real_list.mean(axis=0),recon_list.mean(axis=0))[0][1])
    # print('mmd=',compute_mmd(torch.tensor(real_list), torch.tensor(recon_list)))
    print('mse=',np.mean((real_list-recon_list)**2))
    




if __name__ == "__main__":
    main()



