import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from torchvision import utils  # transforms, models, utils
from Dataset import Dataset
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torchmetrics
from torchmetrics import MetricCollection
import os
import seaborn as sns

class NewBase(pl.LightningModule):

    def __init__(self, params):
        super(NewBase, self).__init__()
        # Sauvegarder les hyperparamètres
        self.save_hyperparameters()
        self.conf = params

        # Définir les transformations d'entrée
        self.input_transforms = None

        # Définir le réseau
        self.net = None

        # Définir les métriques pour l'objet
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanSquaredError(),
            torchmetrics.PeakSignalNoiseRatio(),
            torchmetrics.StructuralSimilarityIndexMeasure(),
            torchmetrics.PearsonCorrCoef()
        ])

        self.metrics_test = torchmetrics.MetricCollection([
            torchmetrics.MeanAbsoluteError(prefix='test_'),
            torchmetrics.MeanSquaredError(prefix='test_'),
            torchmetrics.PeakSignalNoiseRatio(prefix='test_'),
            torchmetrics.StructuralSimilarityIndexMeasure(prefix='test_'),
            torchmetrics.PearsonCorrCoef(prefix='test_')
        ])

    def forward(self, batch):
        raise NotImplementedError('NewBase must be extended by child class that specifies the forward method.')

    def training_step(self, batch, batch_nb):
        inputs, targets = batch

        # Calculer les prédictions
        predictions = self(inputs)

        # Calculer la perte L1
        loss = F.l1_loss(predictions, targets)

        # Log des métriques
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Calculer les prédictions
        predictions = self(inputs)

        # Calculer la perte L1
        loss = F.l1_loss(predictions, targets)

        # Log des métriques
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])

        # Mettre à jour les métriques
        self.metrics.update(predictions, targets)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch

        # Calculer les prédictions
        predictions = self(inputs)

        # Mettre à jour les métriques de test
        self.metrics_test.update(predictions, targets)

    def configure_optimizers(self):
        
        # set net parameters as optimization
        if self.conf['method'] == 'middle_fusion':
            params = [
                {'params': self.fusion_en.parameters()},
                {'params': self.net.parameters()},
            ]
        else:
            params = [
                {'params': self.net.parameters()},
            ]


        optimizer = torch.optim.Adam(
                            params,
                            weight_decay = self.conf['weight_decay'],
                            lr = self.conf['learning_rate'])

        LR_scheduler = {"scheduler": StepLR(optimizer, step_size=7, gamma=0.5), "monitor": "losses/val_total",
                        'name': 'Learning_rate'}
        
        return ([optimizer], [LR_scheduler])
    

    def train_dataloader(self):
        ds = Dataset(self.conf['root_dir'], self.conf['train_csv'], self.conf['pca'], self.train_transforms())
        dl = data.DataLoader(ds, batch_size=self.conf['batch_size'], num_workers=self.conf['num_workers'], shuffle=True)
        return dl
    
    def val_dataloader(self):
        ds = Dataset(self.conf['root_dir'], self.conf['val_csv'], self.conf['pca'], self.val_transforms())
        dl = data.DataLoader(ds, batch_size=self.conf['batch_size'], num_workers=self.conf['num_workers'], shuffle=False)
        return dl
    
    
    def test_dataloader(self):
        ds = Dataset(self.conf['root_dir'], self.conf['test_csv'], self.conf['pca'], self.test_transforms())
        dl = data.DataLoader(ds, batch_size=self.conf['batch_size'], num_workers=self.conf['num_workers'], shuffle=False)
        return dl

    def train_transforms(self):
        return None
    
    def val_transforms(self):
        return None

    def test_transforms(self):
        return None