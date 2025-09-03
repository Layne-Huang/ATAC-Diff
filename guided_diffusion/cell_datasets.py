import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize as sk_normalize
from muon import atac as ac

def normalize(adata, norm_type):
    if norm_type == 'log_norm':
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # sc.pp.scale(adata)
        print('Normalized the dataset use lognorm')
    elif norm_type == 'tfidf':        
        ac.pp.tfidf(adata, scale_factor=1e4)
        sc.pp.log1p(adata)
        # sc.pp.scale(adata)
        # vectorizer = TfidfTransformer()
        # peak_count = adata.X.toarray()
        # vectorizer.fit(peak_count)
        # X = vectorizer.transform(peak_count).toarray()
        # X1 = sk_normalize(X, axis=1, norm='l1')
        # adata.X1 = X1
        print('Normalized the dataset use tf-idf')
    elif norm_type == 'binary':
        adata.X[adata.X > 0] = 1
        print('Binary the dataset')
    else:
        # sc.pp.scale(adata)
        print('No normalization, use raw counts')
    
    return adata

def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder

def load_data(
    *,
    data_dir,
    batch_size,
    ae_dir=None,
    num_gene=0,
    vae=False,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    norm='log_norm',
    denoise=False,
    train=True,
    hidden_dim=128
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_h5ad(data_dir)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    try:
        classes = adata.obs['celltype'].values
    except:
        classes = adata.obs['cell_type'].values
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)
    print(label_encoder.classes_)

    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    batchs = adata.obs['batch'].values
    batch_encoder = LabelEncoder()
    batch_encoder.fit(batchs)
    batchs = batch_encoder.transform(batchs)

    if denoise:
        train_mask = adata.layers['train_mask']
        adata.X[~train_mask] = 0
    else:
        train_mask = None
        
    adata = normalize(adata, norm)

    try:
        cell_data = adata.X.toarray()
    except:
        cell_data = adata.X


    print('Processed dataset shape: {}'.format(adata.shape))
    cell_type = len(list(set(classes)))
    num_gene = cell_data.shape[1]
    # use autoencoder when train diffusion
    cell_emb = None
    if vae:
        autoencoder = load_VAE(ae_dir, num_gene, hidden_dim)
        cell_emb = autoencoder(torch.tensor(cell_data).cuda(),return_latent=True)
        cell_emb = cell_emb.cpu().detach().numpy()
    if not class_cond:
        classes = None
        batchs = None

    dataset = SCDataset(
        cell_data,
        cell_emb=cell_emb,
        classes=classes,
        batchs=batchs,
        train_mask=train_mask
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )
    # if not train:
    # return cell_type, num_gene, loader
    # else:
    #     yield len(list(set(classes)))
    #     yield cell_data.shape[1]
    while True:
        yield from loader



class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict

class SCDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, cell_data, classes, batchs, cell_emb=None,train_mask=None):
        super().__init__()
        self.shape = cell_data.shape
        self.batchs = batchs
        self.cell_types = classes
        self.X_emb = cell_emb
        self.X = cell_data
        # train_mask = adata.layers['train_mask']
        # self.X_cor = self.X.multiply(train_mask).tocsr()
        if train_mask is not None:
            # self.X[~train_mask] = 0
            self.mask = train_mask
        else:
            self.mask = np.ones(self.X.shape, dtype=bool)


    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        mask = self.mask[idx]
        out_dict = {}
        out_dict['latent'] = False
        # out_dict['latent'] = True
        if self.cell_types is not None:
            cell_type = self.cell_types[idx]
            out_dict["cell_type"] = np.array(cell_type, dtype=np.int64)
        if self.batchs is not None:
            batch = self.batchs[idx]
            out_dict["batch"] = np.array(batch, dtype=np.int64)
        if self.X_emb is not None:
            x_emb= self.X_emb[idx]
            out_dict['x_raw'] = x
            return x_emb, out_dict, mask
        
        
        return x, out_dict,mask

