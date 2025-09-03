import torch
from torch.nn import functional as F
import scipy
import numpy as np
from guided_diffusion.misc import as_tensor
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, silhouette_score
from sklearn.cluster import KMeans
import scanpy as sc

def as_tensor(x, assert_type: bool = False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor) and assert_type:
        raise TypeError(f"Expecting tensor or numpy array, got, {type(x)}")
    return x

def masked_rmse(pred, true, mask):
    pred_masked = pred * mask
    true_masked = true * mask
    size = mask.sum()
    return (F.mse_loss(pred_masked, true_masked, reduction='sum') / size).sqrt()


def masked_stdz(x, mask):
    size = mask.sum(1, keepdim=True).clamp(1)
    x = x*mask
    x_ctrd = x - (x.sum(1, keepdim=True) / size)*mask
    # NOTE: multiplied by the factor of sqrt of N
    x_std = x_ctrd.pow(2).sum(1, keepdim=True).sqrt()
    return x_ctrd / x_std


def masked_corr(pred, true, mask):
    pred_masked_stdz = masked_stdz(pred, mask)
    true_masked_stdz = masked_stdz(true, mask)
    corr = (pred_masked_stdz * true_masked_stdz).sum(1).mean()
    return corr

def PearsonCorr(y_pred, y_true):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c, 1)
        / torch.sqrt(torch.sum(y_true_c * y_true_c, 1))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1))
    )
    return pearson

def PearsonCorr1d(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c)
        / torch.sqrt(torch.sum(y_true_c * y_true_c))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c))
    )
    return pearson

def denoising_eval(true, pred, mask):
    true = as_tensor(true, assert_type=True)
    pred = as_tensor(pred, assert_type=True)
    mask = as_tensor(mask, assert_type=True).bool()

    rmse_normed = masked_rmse(pred, true, mask).item()
    corr_normed = masked_corr(pred, true, mask).item()
    global_corr_normed = PearsonCorr1d(pred[mask], true[mask]).item()

    # nonzero_masked = (true > 0) * mask
    # rmse_normed_nonzeros = masked_rmse(pred, true, nonzero_masked).item()
    # corr_normed_nonzeros = masked_corr(pred, true, nonzero_masked).item()

    corr_normed_all = PearsonCorr(pred, true).item()
    rmse_normed_all = F.mse_loss(pred, true).sqrt().item()

    r = scipy.stats.linregress(pred[mask].cpu().numpy(), true[mask].cpu().numpy())[2]
    # r_all = scipy.stats.linregress(pred.ravel().cpu().numpy(), true.ravel().cpu().numpy())[2]


    return {
        'denoise_rmse_normed': rmse_normed,
        'denoise_corr_normed': corr_normed,
        'denoise_global_corr_normed': global_corr_normed,
        'denoise_global_r2_normed': r ** 2,
        # 'denoise_rmse_normed_nonzeros': rmse_normed_nonzeros,
        # 'denoise_corr_normed_nonzeros': corr_normed_nonzeros,
        'denoise_rmse_normed_all': rmse_normed_all,
        'denoise_corr_normed_all': corr_normed_all,
        # 'denoise_global_r2_normed_all': r_all ** 2,
    }

def cluster_eval(adata,rep):
    try:
        true_labels = adata.obs['celltype']
    except:
        true_labels = adata.obs['cell_type']
    if 'kmeans' not in adata.obs:
        try:
            kmenas = KMeans(n_clusters=adata.obs['celltype'].unique().shape[0], n_init=20, random_state=0)
        except:
            kmenas = KMeans(n_clusters=adata.obs['cell_type'].unique().shape[0], n_init=20, random_state=0)
        adata.obs['kmeans'] = kmenas.fit_predict(adata.obsm[rep])

    cluster_labels = adata.obs['kmeans']
    # cluster_lables = feb.obs['leiden']

    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
    print("NMI Score:", nmi_score)

    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    print("ARI Score:", ari_score)

    homogenity = homogeneity_score(true_labels, cluster_labels)
    print("Homogenity Score:", homogenity)

    # silhouette = silhouette_score(adata.obsm['X_umap'], cluster_labels)
    # print('Silhouette Score UMAP', silhouette)

    silhouette = silhouette_score(adata.obsm[rep], cluster_labels)
    print('Silhouette Score', silhouette)

    silhouette = silhouette_score(adata.obsm[rep], true_labels)
    print('Cell Type Silhouette Score', silhouette)


    sc.pp.neighbors(adata, n_neighbors=30, use_rep=rep)
    sc.tl.umap(adata, min_dist=0.1)
    color = [c for c in ['celltype', 'kmeans', 'leiden', 'cell_type','latent','gmm','subtype','label'] if c in adata.obs]

    silhouette = silhouette_score(adata.obsm['X_umap'], cluster_labels)
    print('UMAP Silhouette Score', silhouette)
    sc.pl.umap(adata, color=color, show=False, wspace=0.4, ncols=4)