"""Copyright (c) 2018, Haavard Kvamme
                 2021, Schrod Stefan"""

import os
from functools import partial

import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.model_selection import train_test_split
import pandas as pd
from torch import Tensor
from torch import nn
from model import *
from torch.utils.data import *
import torch.nn.functional as F

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Binarizer

def data_transformer():
    numeric_features = ['RDELAY', "AGE", 'RSBP']
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_features = [ 'RCONSC', 'SEX', 'RSLEEP', 'RATRIAL', 'RCT', 'RVISINF',
        'RHEP24', 'RASP3', 'RDEF1', 'RDEF2', 'RDEF3', 'RDEF4', 'RDEF5',
        'RDEF6', 'RDEF7', 'RDEF8', 'STYPE'] 
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    return Pipeline(steps=[("preprocessor", preprocessor)])



class IHDP(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for PathMnist dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, val_set_fraction=0.2, batch_size=32, num_workers=8, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_set_fraction = val_set_fraction


    def prepare_data(self):
        self.ihdp_data = pd.read_csv('data/IHDP/ihdp_project2.csv.gz', compression='gzip')
        self.X = self.ihdp_data[self.ihdp_data.columns[1:26]].to_numpy()
        self.Y = self.ihdp_data['Y'].to_numpy()
        self.T = self.ihdp_data['T'].to_numpy()
        self.dataset = TensorDataset(torch.Tensor(self.X), torch.Tensor(self.Y),
                            torch.Tensor(self.T))
        

    def setup(self, stage=None):
        X_train, X_test, Y_train, Y_test, treatment_train, treatment_test = train_test_split(self.X, self.Y, self.T, test_size=0.25, random_state=42)

        data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train),
                            torch.Tensor(treatment_train))
        test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test),
                            torch.Tensor(treatment_test))
        test_abs = int(len(X_train) * (1 - self.val_set_fraction))
        train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs])

        self.train = train_subset
        self.val = val_subset
        self.test = test_data

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class IST(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for PathMnist dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, treatment='RXASP', val_set_fraction=0.2, batch_size=32, num_workers=8, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.treatment = treatment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_set_fraction = val_set_fraction
        self.data_transformer = data_transformer()


    def prepare_data(self):
        ist_data = pd.read_csv('data/IST/IST_observational.csv')
        ist_data['Y'] = (ist_data['DDEAD'] == 'Y').astype(int)
        ist_data['T'] = (ist_data[self.treatment] != 'N').astype(int)
        treated_group = ist_data[(ist_data[self.treatment] != 'N')]
        control_group = ist_data[(ist_data[self.treatment] == 'N')]
        self.data = pd.concat([treated_group, control_group])
        self.X = self.data_transformer.fit_transform(self.data)
        self.Y = self.data['Y'].to_numpy()
        self.T = self.data['T'].to_numpy()
        self.dataset =  TensorDataset(torch.Tensor(self.X), torch.Tensor(self.Y), torch.Tensor(self.T))
        

    def setup(self, stage=None):
        X_train, X_test, Y_train, Y_test, treatment_train, treatment_test = train_test_split(self.X, self.Y, self.T, test_size=0.25, random_state=42)

        data = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train),
                            torch.Tensor(treatment_train))
        test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test),
                            torch.Tensor(treatment_test))
        test_abs = int(len(X_train) * (1 - self.val_set_fraction))
        train_subset, val_subset = random_split(data, [test_abs, len(X_train) - test_abs])

        self.train = train_subset
        self.val = val_subset
        self.test = test_data

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * torch.matmul(X, Y.t())
    nx = torch.sum(torch.square(X), dim=1, keepdim=True)
    ny = torch.sum(torch.square(Y), dim=1, keepdim=True)
    D = (C + ny.t()) + nx
    return D

def safe_sqrt(x, eps=1e-6):
    return torch.sqrt(torch.clamp(x, min=eps))

def wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """
    
    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = float(Xc.shape[0])
    nt = float(Xt.shape[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = torch.mean(M)
    M_drop = torch.nn.functional.dropout(M, p=10/(nc*nt))
    delta = torch.max(M).detach()
    eff_lam = lam / M_mean

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * torch.ones((1, M.shape[1]), device=M.device)
    col = torch.cat([delta * torch.ones((M.shape[0], 1), device=M.device), torch.zeros((1, 1), device=M.device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    ''' Compute marginal vectors '''
    a = torch.cat([p * torch.ones((torch.sum(t > 0), 1), device=M.device) / nt, (1 - p) * torch.ones((1, 1), device=M.device)], dim=0)
    b = torch.cat([(1 - p) * torch.ones((torch.sum(t < 1), 1), device=M.device) / nc, p * torch.ones((1, 1), device=M.device)], dim=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (torch.matmul(ainvK, (b / torch.matmul(u.t(), K).t())))
    v = b / (torch.matmul(u.t(), K).t())

    T = u * (v.t() * K)

    if not backpropT:
        T = T.detach()

    E = T * Mt
    D = 2 * torch.sum(E)

    return D, Mlam


def mmd2_lin(X, t, p):
    """ Linear MMD """
    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.mean(Xc, dim=0)
    mean_treated = torch.mean(Xt, dim=0)

    mmd = torch.sum(torch.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))

    return mmd

    

Wasserstein = 0
SquaredLinearMMD=1
class CFRNet_Loss(torch.nn.Module):
    """Loss function for the Tar net structure
    uses the same Cox Loss as PyCox but separates between different treatment classes
    """
    # alpha = 0 TARNet
    # alpha > 0 CFR
    # Wasserstein 
    def __init__(self, alpha=0, mode=Wasserstein):
        self.alpha = alpha
        self.mode = mode
        super().__init__()

    def forward(self, log_h: Tensor, out:Tensor, durations: Tensor, treatments: Tensor) -> Tensor:
        mask0 = treatments == 0
        mask1 = treatments == 1

        mse=nn.MSELoss()

        loss_t0 = mse(log_h[mask0], durations[mask0])
        loss_t1 = mse(log_h[mask1], durations[mask1])

        """Imbalance loss"""
        p = torch.sum(mask1) / treatments.shape[0]
        if self.alpha == 0.0:
            imbalance_loss = 0.0
        else:
            if self.mode == Wasserstein:
                imbalance_loss, _ = wasserstein(out, treatments, p)
            else:
                imbalance_loss = mmd2_lin(out, treatments, p)
        return (1.0 - p) * loss_t0 + p * loss_t1 + self.alpha * imbalance_loss

from sklearn.metrics import mean_absolute_error
def get_ITE_CFRNet(model, X, treatment, true_ITE=None, best_treatment=None):

    pred,_ = model.predict_numpy(X, treatment)
    pred_cf,_ = model.predict_numpy(X, 1-treatment)

    ITE = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if treatment[i] == 0:
            ITE[i] = pred_cf[i]-pred[i]
        else:
            ITE[i] = pred[i]-pred_cf[i]


    correct_predicted_probability=None
    if best_treatment is not None:
        correct_predicted_probability=np.sum(best_treatment==(ITE>0)*1)/best_treatment.shape[0]
        print('Fraction best choice: ' + str(correct_predicted_probability))

    if true_ITE is not None:
        print('Mean absolute error: ', mean_absolute_error(true_ITE, ITE))
    
    print('Average treatment effect: ', ITE.mean())

    return ITE, correct_predicted_probability


import seaborn as sns
def plot_ITE_correlation(pred_ITE, y_true, y_cf, treatment):
    ITE = np.zeros(pred_ITE.shape[0])
    true_ITE0 = -(y_true[treatment == 0] - y_cf[treatment == 0])
    true_ITE1 = y_true[treatment == 1] - y_cf[treatment == 1]
    k, j = 0, 0
    for i in range(pred_ITE.shape[0]):
        if treatment[i] == 0:
            ITE[i] = true_ITE0[k]
            k = k + 1
        else:
            ITE[i] = true_ITE1[j]
            j = j + 1

    ax=sns.scatterplot(x=ITE,y=pred_ITE)
    ax.set(xlabel='ITE', ylabel='pred_ITE')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_TSNE(model, dataset_name, X, treatment, method_name):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=42)
    tsne_original_results = tsne.fit_transform(X)

    fig, ax = plt.subplots()
    for i, color in enumerate(['tab:blue', 'tab:orange']):
        x=tsne_original_results[treatment==i][:,0]
        y=tsne_original_results[treatment==i][:,1]

        ax.scatter(x, y, c=color, label='treated' if i==1 else 'control',
                alpha=0.3, edgecolors='none')

    # produce a legend with the unique colors from the scatter
    ax.legend()
    plt.title('tNSE Visulization before balancing')
    plt.savefig(dataset_name+'_original_TNSE.png')

    _, transformed_X = model.predict_numpy(X, treatment)
    tsne_transformed = TSNE(n_components=2, n_iter=300)
    tsne_results = tsne_transformed.fit_transform(transformed_X)

    fig, ax = plt.subplots()
    for i, color in enumerate(['tab:blue', 'tab:orange']):
        x=tsne_results[treatment==i][:,0]
        y=tsne_results[treatment==i][:,1]

        ax.scatter(x, y, c=color, label='treated' if i==1 else 'control',
                alpha=0.3, edgecolors='none')

    # produce a legend with the unique colors from the scatter
    ax.legend()
    plt.title('tNSE Visulization after balancing using ' + method_name)
    plt.savefig(dataset_name + '_' + method_name + '_transformed_TNSE.png')
