"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import random

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from VAE.VAE import Latent_VAE
def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main():
    data_path = '/om/user/layne_h/project/atacdiff/output/forebrain/mu_x0emb.npz'
    
    real_x0_emb_sample = np.load(data_path,allow_pickle=True)['x0_embs']
    cond_mean = th.tensor(np.mean(real_x0_emb_sample, axis=0)).to(device=dist_util.dev())
    cond_std = th.tensor(np.std(real_x0_emb_sample, axis=0)).to(device=dist_util.dev())
    # x0_embs = np.load('/om/user/layne_h/project/atacdiff/output/forebrain/real_x0_emb.npz')['x0_embs']
    # x0_embs = np.load('/om/user/layne_h/project/atacdiff/output/forebrain/forebrain_mu_x0emb.npz')['x0_embs']
    real_x0_emb_sample = np.load('/om/user/layne_h/project/atacdiff/output/buen/diff_x0_emb_32.npz')['cell_gen']
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='checkpoint/sample_logs')

    logger.log("creating model and diffusion...")
    args.cell_types = 19#8,1
    args.input_dim = 128#106642#18996#15099
    args.raw_dim = args.num_genes
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # latnet_vae = Latent_VAE(dims=[128,256,256,64],data_path=data_path,binary=False).to(dist_util.dev())
    # latnet_vae.load_state_dict(th.load(args.lt_vae)[0])
    logger.log("sampling...")
    all_cells = []
    while len(all_cells) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = 1*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            model_kwargs["cell_type"] = classes
            model_kwargs["batch"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # model_kwargs["latent"] = [True]
        # with th.no_grad():
        #     x_emb = latnet_vae.get_x_emb(args.batch_size, dist_util.dev())
        #     # x_emb = x_emb * cond_std + cond_mean
        x_emb = th.tensor(real_x0_emb_sample[1000:1000+args.batch_size]).to(dist_util.dev())
        sample, traj, _ = sample_fn(
            model,
            (args.batch_size, args.input_dim), 
            x_input = x_emb,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=1000
            # noise=th.tensor(sc.read_h5ad('masked_cell.h5ad').X, device=dist_util.dev()),
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_cells.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_cells) * args.batch_size} samples")

    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, traj, args.sample_dir)

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=9631, #2034, #2088,
        batch_size=9631, #2034, #2088,
        num_genes= 106642, #103151, #103151, #11285,
        use_ddim=False,
        # model_path='/om/user/layne_h/project/atacdiff/checkpoint/forebrain/tf_vae_lognorm_forebrain/model010000.pt',
        # model_path = '/om/user/layne_h/project/atacdiff/checkpoint/forebrain/lognorm_mu/model010000.pt',
        # model_path = '/om/user/layne_h/project/atacdiff/checkpoint/forebrain/tf_vae_lognorm_forebrain_nox0emb/model010000.pt',
        # model_path = '/om/user/layne_h/project/ata cdiff/checkpoint/forebrain_wo_mi/x0_emb_dim32_mean_no_binary/model010008.pt',
        # model_path = '/om/user/layne_h/project/atacdiff/checkpoint/buen_wo_mi/x0_emb_dim32_mean_no_binary/model010000.pt',
        model_path = '/om/user/layne_h/project/atacdiff/checkpoint/pbmc10k_wo_mi/x0_emb_dim32_mean_no_binary/model010032.pt',
        data_dir="/",
        sample_dir="/om/user/layne_h/project/atacdiff/output/pbmc10k_wo_mi/diff_x0_x0emb_32",
        lt_vae = '/om/user/layne_h/project/atacdiff/checkpoint/pbmc10k_wo_mi/Latent_VAE_x0_mu_128.pt'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True # 设置随机数种子
    # th.backends.cudnn.enabled = False


if __name__ == "__main__":
    main()
