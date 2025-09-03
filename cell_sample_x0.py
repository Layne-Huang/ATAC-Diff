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
from guided_diffusion.cell_datasets import load_data
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
    setup_seed(1234)
    args = create_argparser().parse_args()
    args.class_cond = False

    cell_type, input_dim, data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        ae_dir=args.ae_dir,
        # num_gene=args.num_genes,
        deterministic=True,
        denoise=True,
        train=False,
        norm='log_norm',
        vae=args.vae)

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

    # latnet_vae = Latent_VAE(dims=[256,256,256,64],binary=False).to(dist_util.dev())
    # latnet_vae.load_state_dict(th.load(args.lt_vae)[0])
    logger.log("sampling...")
    all_cells = []
    all_x0_emb = []
    for data,cond,mask in data_loader:
        model_kwargs = {}
        if args.class_cond:
            classes = 1*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            model_kwargs["cell_type"] = classes
            model_kwargs["batch"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # with th.no_grad():
        #     x_emb = latnet_vae.get_x_emb(args.batch_size, dist_util.dev())
        sample, traj, x0_emb = sample_fn(
            model,
            (args.batch_size, args.input_dim), 
            x_input = data.to(dist_util.dev()),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=1000
            # noise=th.tensor(sc.read_h5ad('masked_cell.h5ad').X, device=dist_util.dev()),
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_cells.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_x0_emb = [th.zeros_like(x0_emb) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_x0_emb, x0_emb)
        all_x0_emb.extend([x0.cpu().numpy() for x0 in gathered_x0_emb])
        logger.log(f"created {len(all_cells) * args.batch_size} samples")

    # arr = np.concatenate(all_cells, axis=0)
    # save_data(arr, traj, args.sample_dir)

    x0_embs = np.concatenate(all_x0_emb, axis=0)
    np.savez(args.sample_dir + '_x0emb', x0_embs=x0_embs)

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    # dm = model_and_diffusion_defaults()
    defaults = dict(
        clip_denoised=False,
        num_samples=2088,
        batch_size=2088,
        num_genes=11285,
        class_cond=False,
        use_ddim=False,
        vae=False,
        model_path="/om/user/layne_h/project/atacdiff/checkpoint/forebrain/x0_emb_128/model002000.pt",
        ae_dir = '/om/user/layne_h/project/scDiffusion/checkpoint/AE/forebraim_AE_lgnorm1e3/model_seed=0_step=9999.pt',
        data_dir='/om/user/layne_h/ATAC_data/Forebrain.h5ad',
        sample_dir="/om/user/layne_h/project/atacdiff/output/forebrain/x0_emb_128",
        # lt_vae = '/om/user/layne_h/project/atacdiff/checkpoint/forebrain/Latent_VAE.pt'
        lt_vae = '/om/user/layne_h/project/atacdiff/checkpoint/forebrain/Latent_VAE_x0.pt'
    )
    defaults.update(model_and_diffusion_defaults())

    # dm.update(defaults)
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
