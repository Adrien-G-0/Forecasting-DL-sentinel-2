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
import json
from NewArchitectures import NewArchitectures
from NewDataset import NewDatasetGlobal
from torch.utils.data import DataLoader


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

    def training_step(self, batch, batch_idx):
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
    

    def _finalize_epoch(self, metrics, prefix):
        """
        Calcule et loggue les métriques à la fin d’un epoch, avec enregistrement dans TensorBoard.
        
        Args:
            metrics (torchmetrics.MetricCollection): Les métriques à évaluer.
            prefix (str): "train", "val", ou "test".
        """
        results = metrics.compute()
        
        # Log Lightning (standard)
        for name, value in results.items():
            metric_name = f"{prefix}/{name}"
            self.log(metric_name, value, prog_bar=(prefix != "test"), on_epoch=True, sync_dist=True)

            # Log explicite pour TensorBoard (add_scalar)
            if isinstance(value, torch.Tensor):
                value = value.item()
            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.add_scalar(metric_name, value, self.current_epoch)

        metrics.reset()

    def training_epoch_end(self, outputs):
        self._finalize_epoch(self.train_metrics, "train")

    def validation_epoch_end(self, outputs):
        self._finalize_epoch(self.val_metrics, "val")

    def test_epoch_end(self, outputs):
        self._finalize_epoch(self.test_metrics, "test")



    def train_transforms(self):
        return None
    
    def val_transforms(self):
        return None

    def test_transforms(self):
        return None






def main():
    # ---------------------
    # 1. Charger la config
    # ---------------------
    with open("params.json", "r") as f:
        params = json.load(f)

    # ---------------------
    # 2. Initialiser le modèle
    # ---------------------
    model = NewArchitectures(params)

    # ---------------------
    # 3. Créer les DataLoaders
    # ---------------------
    train_set = NewDatasetGlobal(split="train", transforms=model.input_transforms)
    val_set = NewDatasetGlobal(split="val", transforms=model.input_transforms)

    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False, num_workers=4)

    # ---------------------
    # 4. Logger & Callbacks
    # ---------------------
    logger = TensorBoardLogger(save_dir="logs", name=params.get("experiment_name", "default"))

    checkpoint_callback = ModelCheckpoint(
        monitor="val/MeanAbsoluteError",  # ou une autre métrique pertinente
        save_top_k=1,
        mode="min",
        filename="best-checkpoint"
    )
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val/MeanSquaredError",  # Surveille la même métrique que dans ModelCheckpoint
        patience=5,  # Nombre d'epochs sans amélioration avant d'arrêter
        mode="min",  # "min" car on veut minimiser l'erreur
        verbose=True  # Affiche les informations d'arrêt
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ---------------------
    # 5. Trainer
    # ---------------------
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        max_epochs=params["max_epochs"],
        accelerator="gpu" if params.get("use_gpu", True) else "cpu",
        devices=1,
        log_every_n_steps=10
    )


    # ---------------------
    # 6. Lancer l'entraînement
    # ---------------------
    trainer.fit(model, train_loader, val_loader)

    # ---------------------
    # 7. Évaluation finale
    # ---------------------
    test_set = NewDatasetGlobal(split="test", transforms=model.input_transforms)
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False, num_workers=4)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()