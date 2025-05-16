import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
# from torchvision import utils  # transforms, models, utils

from NewDataset import Dataset
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torchmetrics
import argparse,json




class Base(pl.LightningModule):

    def __init__(self, params):
        super(Base, self).__init__()
        # Sauvegarder les hyperparamètres
        self.save_hyperparameters()
        self.conf = params

        # Définir les transformations d'entrée
        self.input_transforms = None

        # Définir le réseau
        self.net = None

        # Définir les métriques pour l'objet
        self.metrics = torchmetrics.MetricCollection({
            'mae': torchmetrics.MeanAbsoluteError(),
            'mse': torchmetrics.MeanSquaredError(),
            'psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure(),
            'pearson': torchmetrics.PearsonCorrCoef()
        })
        self.metrics_test = torchmetrics.MetricCollection({
            'mae': torchmetrics.MeanAbsoluteError(),
            'mse': torchmetrics.MeanSquaredError(),
            'psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure(),
            'pearson': torchmetrics.PearsonCorrCoef()
        }, prefix='test_')

    def forward(self, batch):
        raise NotImplementedError('Base must be extended by child class that specifies the forward method.')

    def training_step(self, batch, batch_nb):
        inputs, targets,_ = batch
        
        predictions = self(inputs)
        
        loss = F.l1_loss(predictions, targets)
        # Log des métriques
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _= batch
        # Calculer les prédictions
        predictions = self(inputs)


        # Calculer la perte L1
        loss = F.l1_loss(predictions, targets)
        # Log des métriques
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        # Mettre à jour les métriques
        self.metrics['mae'].update(predictions, targets)
        self.metrics['mse'].update(predictions, targets)
        self.metrics['psnr'].update(predictions, targets)
        self.metrics['ssim'].update(predictions, targets) 

        # Ajustements des dimensions pour certaines métriques
        predictions_flat = predictions.view(-1)  # Aplatit pour Pearson
        targets_flat = targets.view(-1)         # Aplatit pour Pearson
         
        #to avoid error when the variance is too low
        if predictions_flat.var() > 1e-3 :
            self.metrics['pearson'].update(predictions_flat, targets_flat)  # Données applaties
        else:
            # Créer des données avec corrélation nulle (perpendiculaires)
            dummy_x = torch.tensor([0.0, 1.0], device=predictions_flat.device)
            dummy_y = torch.tensor([1.0, 1.0], device=predictions_flat.device)  # Perpendiculaire à dummy_x
            self.metrics['pearson'].update(dummy_x, dummy_y)  # Donnera une corrélation de 0.0



    def on_validation_epoch_end(self):
        # Calculer et logger les métriques
        metrics = self.metrics.compute()
        for key, value in metrics.items():
            self.log(f"val_{key}", value, prog_bar=True)
        
        # Réinitialiser les métriques
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        inputs, targets,_ = batch


        predictions = self(inputs)

        # Mettre à jour les métriques de test
        # self.metrics_test.update(predictions, targets)
        # Mettre à jour les métriques
        self.metrics_test['mae'].update(predictions, targets)
        self.metrics_test['mse'].update(predictions, targets)
        self.metrics_test['psnr'].update(predictions, targets)
        self.metrics_test['ssim'].update(predictions, targets) 

        # Ajustements des dimensions pour certaines métriques
        predictions_flat = predictions.view(-1)  # Aplatit pour Pearson
        targets_flat = targets.view(-1)         # Aplatit pour Pearson
         
        #to avoid error when the variance is too low
        if predictions_flat.var() > 1e-3 :
            self.metrics_test['pearson'].update(predictions_flat, targets_flat)  # Données applaties
        else:
            # Créer des données avec corrélation nulle (perpendiculaires)
            dummy_x = torch.tensor([0.0, 1.0], device=predictions_flat.device)
            dummy_y = torch.tensor([1.0, 1.0], device=predictions_flat.device)  # Perpendiculaire à dummy_x
            self.metrics_test['pearson'].update(dummy_x, dummy_y)  # Donnera une corrélation de 0.0



    def on_test_epoch_end(self):
        # Calculer et logger les métriques de test
        metrics = self.metrics_test.compute()
        for key, value in metrics.items():
            self.log(key, value)
        
        # Réinitialiser les métriques
        self.metrics_test.reset()

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
        ds = Dataset(root_dir=self.conf['root_dir'],sources=self.conf['sources'], split='train', pca=self.conf['pca'], trans=self.train_transforms())
        dl = data.DataLoader(ds, batch_size=self.conf['batch_size'], num_workers=self.conf['num_workers'], shuffle=True)
        return dl
    
    def val_dataloader(self):
        ds = Dataset(root_dir=self.conf['root_dir'],sources=self.conf['sources'], split='val', pca=self.conf['pca'], trans=self.val_transforms())
        dl = data.DataLoader(ds, batch_size=self.conf['batch_size'], num_workers=self.conf['num_workers'], shuffle=False)
        return dl
    
    
    def test_dataloader(self):
        ds = Dataset(root_dir=self.conf['root_dir'],sources=self.conf['sources'], split='test', pca=self.conf['pca'], trans=self.test_transforms())
        dl = data.DataLoader(ds, batch_size=self.conf['batch_size'], num_workers=self.conf['num_workers'], shuffle=False)
        return dl

    def train_transforms(self):
        return None
    
    def val_transforms(self):
        return None

    def test_transforms(self):
        return None
    


    @staticmethod
    def main(cur_class):
        # define parser
        parser = argparse.ArgumentParser()
        # define arguments
        parser.add_argument("-json", "--json", help="Json (used only when training).",
                            default='', type=str)
        parser.add_argument("-ckp", "--checkpoint", help="Checkpoint (used only when regen/testing).",
                            default='', type=str)
        parser.add_argument("-out", "--out", help="Output filename.",
                            default='', type=str)
        parser.add_argument("-test", "--test", help="If set, computes metrics instead of regen.",
                            action='store_true')
        # parse args
        args = parser.parse_args()

        # check if we are in training or testing
        if not args.test:
            # if checkpoint not specified, load from json
            if args.checkpoint == '':
                # read json
                print(args.json)
                with open(args.json) as f:
                    conf = json.load(f)
                # init model
                model = cur_class(conf)
            else:
                # load from checkpoint
                model = cur_class.load_from_checkpoint(args.checkpoint)
                conf = model.conf

            # show model
            print(model)

            
            # Define callbacks
            callbacks = [
                RichProgressBar(),
                ModelCheckpoint(
                    monitor='val_mae', 
                    mode='min', 
                    save_top_k=1, 
                    save_last=True, 
                    filename='l1_loss'
                ),
                LearningRateMonitor(logging_interval='epoch'),
            ]
            
            # Early Stopping
            early_stop_callback = EarlyStopping(
                monitor='val_mae',         # Metric to observe
                min_delta=0.00,            # minimum change to consider an improvement
                patience=4,                # number of epochs without improvement before stopping
                verbose=True,              # print messages of early stopping
                mode='min',                # minimizing the metric
                strict=True,               # if True, observed metric must be present at each epoch
            )
            
            # Define trainer
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=1,
                max_epochs=conf.get('n_epochs', 100),  # Use get() for safer dictionary access
                num_sanity_val_steps=2 if args.checkpoint == '' else 0,
                logger=TensorBoardLogger('checkpoints', name=conf['experiment_name']),
                profiler="simple",
                callbacks=[*callbacks, early_stop_callback]
                
            )
            
            # Train the model
            if args.checkpoint == '':
                trainer.fit(model)
            else:
                trainer.fit(model, ckpt_path=args.checkpoint)
                
            # Test after training
            trainer.test(model)
        else:
            # Test mode only
            
            model = cur_class.load_from_checkpoint(args.checkpoint)
            conf = model.conf
            tester = pl.Trainer()
            tester.test(model)
            
            

if __name__ == '__main__':

    from NewArchitectures import NewArchitectures
    if True:
        with open('params.json') as f:
            conf = json.load(f)
        model= NewArchitectures(conf)
            # 5. Définissez les callbacks
        callbacks = [
            RichProgressBar(),
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=True,
                filename='l1_loss-{epoch:02d}-{total:.4f}'
            ),
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(
                monitor='val_mae',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='min',
                strict=True,
            )
        ]

        # 6. Créez le Trainer
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=conf['n_epochs'],
            num_sanity_val_steps=2,
            logger=TensorBoardLogger('checkpoints', name=conf['experiment_name']  + "_".join(conf['sources']+ ["_"] + [conf['method']])),
            callbacks=callbacks
        )
        # 7. Lancez l'entraînement
        trainer.fit(model)
        # conf = model.conf
        tester = pl.Trainer()
        tester.test(model)
    if False:
        # load from checkpoint
        ckp="checkpoints/dem_sar/version_0/checkpoints/l1_loss-epoch=15-total=0.0000.ckpt"
        model = NewArchitectures.load_from_checkpoint(ckp)
        conf = model.conf
        tester = pl.Trainer()
        tester.test(model)

