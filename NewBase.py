import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F


class NewBase(pl.LightningModule):
    def __init__(self, params):
        super(NewBase, self).__init__()
        # save params
        self.save_hyperparameters()
        self.conf = params
        # define transforms
        self.input_transforms = None
        # define network
        self.net = None

    def forward(self, x):
        raise NotImplementedError('Base must be extended by child class that specifies the forward method.')

    def training_step(self, batch, batch_nb):
        inputs, targets, _ = batch # for the name of the image

        # append
        logits = self(inputs)
        gt_lc, gt_sau = targets

        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)