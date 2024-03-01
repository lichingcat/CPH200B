"""Copyright (c) 2018, Haavard Kvamme
                 2021, Schrod Stefan"""
import os
import numpy as np
from torch import nn
import numpy as np
import torch
from torch.utils.data import *

import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from utils import *

class Classifer(pl.LightningModule):
    def __init__(self, init_lr=1e-4, alpha=0, loss_mode=0):
        super().__init__()
        self.save_hyperparameters()
        self.init_lr = init_lr

        self.mse = torchmetrics.MeanSquaredError()

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

        self.loss_fkt = CFRNet_Loss(alpha=alpha, mode=loss_mode)
        self.loss_fkt_val = CFRNet_Loss(alpha=alpha, mode=loss_mode)

    def training_step(self, batch, batch_idx):
        x, y, treatment = batch
        y_pred = self.forward(x, treatment)

        # forward + backward + optimize
        loss = self.loss_fkt(y_pred[0], y_pred[1], y, treatment)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_pred,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val, treatment_val = batch
        # forward + backward + optimize
        y_pred_val = self.forward(x_val, treatment_val)

        loss_val = self.loss_fkt_val(y_pred_val[0], y_pred_val[1], y_val, treatment_val)
        self.log("val_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_outputs.append({
            "y_hat": y_pred_val,
            "y": y_val
        })

    def test_step(self, batch, batch_idx):
        x, y, treatment = batch
        # forward + backward + optimize
        y_pred = self.forward(x, treatment)

        loss = self.loss_fkt_val(y_pred[0], y_pred[1], y, treatment)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        self.test_outputs.append({
            "y_hat": y_pred,
            "y": y
        })

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return self.opt
    
import torch.nn.functional as F
class ExpLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ExpLinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.elu(self.linear(x))
        return x

class ExpLinearBlock(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(ExpLinearBlock, self).__init__()
        self.layer1 = ExpLinearLayer(input_size, hidden_size)
        self.layer2 = ExpLinearLayer(hidden_size, hidden_size)
        self.layer3 = ExpLinearLayer(hidden_size, out_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class CFRNet(Classifer):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(self, in_features=25, hidden_size_shared=200, hidden_size_indiv=100, out_features=1,
                 num_treatments=1, batch_norm=True, dropout=None, init_lr=1e-4, alpha=0, loss_mode=0, **kwargs):
        super().__init__(init_lr=init_lr, alpha=alpha, loss_mode=loss_mode)
        self.save_hyperparameters()
        self.shared_net = ExpLinearBlock(in_features, hidden_size_shared, hidden_size_shared)
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_treatments+1):
            net = ExpLinearBlock(hidden_size_shared, hidden_size_indiv, out_features)
            self.risk_nets.append(net)

    def forward(self, input, treatment):
        # treatment=input[:,-1]
        N = treatment.shape[0]
        y = torch.zeros_like(treatment)

        out = self.shared_net(input)

        out0 = out[treatment == 0]
        out1 = out[treatment == 1]

        out0 = self.risk_nets[0](out0)
        out1 = self.risk_nets[1](out1)

        k, j = 0, 0
        for i in range(N):
            if treatment[i] == 0:
                y[i] = out0[k]
                k = k + 1
            else:
                y[i] = out1[j]
                j = j + 1

        return y, out

    def predict(self, X, treatment):
        self.eval()
        out = self(X, treatment)
        self.train()
        return out[0], out[1]

    def predict_numpy(self, X, treatment):
        self.eval()
        X = torch.Tensor(X)
        treatment = torch.Tensor(treatment)
        out = self(X, treatment)
        self.train()
        return out[0].detach(), out[1].detach()