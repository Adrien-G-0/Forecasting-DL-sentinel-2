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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torchmetrics
import argparse,json
import matplotlib.pyplot as plt
import weightwatcher as ww

import autres




class Base(pl.LightningModule):

    def __init__(self, params):
        super(Base, self).__init__()
        # Save hyper parameters
        self.save_hyperparameters()
        self.conf = params

        # Define the transformations
        self.input_transforms = None

        # Define the network architecture
        self.net = None

        # Define the metrics
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

        self.distribution_loss = torch.empty(0, device='cpu', dtype=torch.float32)

    def forward(self, batch):
        raise NotImplementedError('Base must be extended by child class that specifies the forward method.')

    def training_step(self, batch, batch_nb):
        inputs, targets,_ = batch
        
        predictions = self(inputs)
        
        loss = F.l1_loss(predictions, targets)
        # masked_loss= self.masked_lc_loss(loss, inputs, self.conf['sources'])
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _= batch
        
        predictions = self(inputs)

        loss = F.l1_loss(predictions, targets)
        # masked_loss= self.masked_lc_loss(loss, inputs, self.conf['sources'], alpha=0)


        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])

        # Update each metrics
        self.metrics['mae'].update(predictions, targets)
        self.metrics['mse'].update(predictions, targets)
        self.metrics['psnr'].update(predictions, targets)
        self.metrics['ssim'].update(predictions, targets) 

        # Pearson correlation need to be flattened before computation
        predictions_flat = predictions.view(-1)  
        targets_flat = targets.view(-1)        
         

        # avoid warning when variance is too low
        if predictions_flat.var() > 1e-3 :
            self.metrics['pearson'].update(predictions_flat, targets_flat) 
        else:
            # Creating orthogonal data when the variance is too low
            dummy_x = torch.tensor([0.0, 1.0], device=predictions_flat.device)
            dummy_y = torch.tensor([1.0, 1.0], device=predictions_flat.device)  
            self.metrics['pearson'].update(dummy_x, dummy_y)  



    def on_validation_epoch_end(self):
        
        metrics = self.metrics.compute()
        for key, value in metrics.items():
            self.log(f"val_{key}", value, prog_bar=True)
        
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        inputs, targets,_ = batch


        predictions = self(inputs)

        # Update metrics as in validation step
        loss = F.l1_loss(predictions, targets)
        # masked_loss= self.masked_lc_loss(loss, inputs, self.conf['sources'], alpha=0)


        self.log("test_loss", loss, prog_bar=False, batch_size=self.conf['batch_size'])

        
        self.metrics_test['mae'].update(predictions, targets)
        self.metrics_test['mse'].update(predictions, targets)
        self.metrics_test['psnr'].update(predictions, targets)
        self.metrics_test['ssim'].update(predictions, targets)
        
        predictions_flat = predictions.view(-1)  
        targets_flat = targets.view(-1)   

        if batch_idx in [0,20]:  # Enregistrer seulement pour le premier batch

            # self.logger.experiment.add_images('test_inputs', torch.cat(inputs,dim=1), batch_idx)   
            self.logger.experiment.add_images('test_targets', targets, batch_idx)   
            self.logger.experiment.add_images('test_predictions', predictions, batch_idx)             

        # distribution of loss per image
        loss=  F.l1_loss(predictions, targets, reduction='none')  
        loss_per_image = loss.mean(dim=(1, 2, 3)).cpu() # Average loss per image
        self.distribution_loss=torch.cat((self.distribution_loss, loss_per_image), dim=0)  # Concatenate losses

         
        
        if predictions_flat.var() > 1e-3 :
            self.metrics_test['pearson'].update(predictions_flat, targets_flat)  
        else:
            dummy_x = torch.tensor([0.0, 1.0], device=predictions_flat.device)
            dummy_y = torch.tensor([1.0, 1.0], device=predictions_flat.device)  
            self.metrics_test['pearson'].update(dummy_x, dummy_y) 



    def on_test_epoch_end(self):
        # Same as on validation_epoch_end
        metrics = self.metrics_test.compute()
        for key, value in metrics.items():
            self.log(key, value)
        
        self.metrics_test.reset()

        #log the ditribution of loss
        self.logger.experiment.add_histogram('loss_distribution', self.distribution_loss, self.current_epoch)



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

        Plateau_scheduler = {"scheduler":ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=8),
                            "monitor":"val_loss",
                            "interval":"epoch",
                            "name":"ReduceLROnPlateau"}

        return {
                "optimizer": optimizer,
                "lr_scheduler": Plateau_scheduler 
                }        

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
    



    def ww_alpha_values_plot(self):
        
        """
        Génère les valeurs alpha de WeightWatcher
        
        Args:
            model: Le modèle PyTorch
            return_figure: Si True, retourne la figure matplotlib, sinon les valeurs alpha
        
        Returns:
            list ou matplotlib.figure.Figure: Valeurs alpha ou figure selon return_figure
        """
        # Initialiser WeightWatcher
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze()
        
        # Préparer les données
        alpha_values = details.alpha.tolist()
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histogramme des valeurs alpha
        ax.hist(alpha_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Valeurs Alpha")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution des valeurs Alpha (Weight Watcher)")
        
        # Lignes de référence
        ax.axvline(x=2, color="r", linestyle="--", label="Seuil surentraîné (α = 2)")
        ax.axvline(x=6, color="orange", linestyle="--", label="Seuil sous-entraîné (α = 6)")
        
        # Ajouter une légende
        ax.legend()
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
        

    def  masked_lc_loss(self,loss,inputs,sources,alpha=0):
        '''
        Creating a mask that only keep region of interest (vegetation, crops...)
        '''
        if 'lc' not in sources:
            return loss.mean()
        idx_lc = self.conf['sources'].index('lc') 
        lc=inputs[idx_lc].argmax(dim=1) #get the class, opposite operatin of onehot encoding

        L=[5,6,7] # classe for forest, vegetation and water
        L_tensor = torch.tensor(L, device=inputs.device)

         # Créer un masque booléen pour les classes d'intérêt
        mask = (lc.unsqueeze(-1) == L_tensor.view(1, 1, 1, -1)).any(dim=-1)

        # Redimensionner le masque pour correspondre à la perte
        mask = mask.unsqueeze(1).float()  # Shape: (batch_size, 1, img_size, img_size)

        # Appliquer le masque à la perte
        masked_loss = loss * mask + alpha * (1 - mask)
        return masked_loss.mean()



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
                monitor='val_loss',         # Metric to observe
                min_delta=0.00,             # minimum change to consider an improvement
                patience=10,                # number of epochs without improvement before stopping
                verbose=True,               # print messages of early stopping
                mode='min',                 # minimizing the metric
                strict=True,                # if True, observed metric must be present at each epoch
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
            # 5.callbacks
        callbacks = [
            RichProgressBar(),
            ModelCheckpoint(
                monitor='val_mae',
                mode='min',
                save_top_k=1,
                save_last=True,
                filename='l1_loss-{epoch:02d}-{val_loss:.4f}'
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

        # 6. Trainer
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=conf['n_epochs']*0,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger('checkpoints', name='test/'+conf['experiment_name']  + "_".join(conf['sources'] + [conf['method']])),
            callbacks=callbacks
        )
        # 7. Training and test
        # trainer.fit(model)

        trainer.test(model)
    if False:
        # load from checkpoint
        ckp=""
        model = NewArchitectures.load_from_checkpoint(ckp)
        conf = model.conf
        tester = pl.Trainer()
        tester.test(model)

