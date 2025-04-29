import argparse
import json
import matplotlib
import torchmetrics.segmentation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from einops import rearrange, repeat  # , reduce
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
from pytorch_lightning import seed_everything

class Base(pl.LightningModule):

    def __init__(self, params):
        super(Base, self).__init__()
        # save params
        self.save_hyperparameters()
        self.conf = params
        # define transforms
        self.input_transforms = None
        # define network
        self.net = None

         # define common arguments
        common_args_macro_lc = {'average': 'macro', 'num_classes': self.conf['n_classes_lc']}   #, 'ignore_index':0}, 'device':self.device}   
        common_args_micro_lc = {'average': 'micro', 'num_classes': self.conf['n_classes_lc']}   #, 'ignore_index':0}, 'device':self.device}  

        common_args_macro_sau = {'average': 'macro', 'num_classes': self.conf['n_classes_sau']} #, 'ignore_index':0} 'device':self.device}  
        common_args_micro_sau = {'average': 'micro', 'num_classes': self.conf['n_classes_sau']} #, 'ignore_index':0} 'device':self.device}  
        
        # define metrics for object
        metrics_lc = MetricCollection({
            'macro/accuracy': torchmetrics.Accuracy(task="multiclass", **common_args_macro_lc),
            'macro/iou': torchmetrics.JaccardIndex(task="multiclass", **common_args_macro_lc),
            'macro/dice': torchmetrics.segmentation.DiceScore(**common_args_macro_lc),
            'macro/precision': torchmetrics.Precision(task='multiclass', **common_args_macro_lc, top_k=1),
            'macro/recall': torchmetrics.Recall(task='multiclass', **common_args_macro_lc, top_k=1),
            'micro/accuracy': torchmetrics.Accuracy(task="multiclass", **common_args_micro_lc),
            'micro/iou': torchmetrics.JaccardIndex(task="multiclass", **common_args_micro_lc),
            'micro/dice': torchmetrics.segmentation.DiceScore(**common_args_micro_lc),
            'micro/precision': torchmetrics.Precision(task='multiclass', **common_args_micro_lc, top_k=1),
            'micro/recall': torchmetrics.Recall(task='multiclass', **common_args_micro_lc, top_k=1),
            })

        metrics_sau = MetricCollection({
            'macro/accuracy': torchmetrics.Accuracy(task="multiclass", **common_args_macro_sau),
            'macro/iou': torchmetrics.JaccardIndex(task="multiclass", **common_args_macro_sau),
            'macro/dice': torchmetrics.segmentation.DiceScore(**common_args_macro_sau),
            'macro/precision': torchmetrics.Precision(task='multiclass', **common_args_macro_sau, top_k=1),
            'macro/recall': torchmetrics.Recall(task='multiclass', **common_args_macro_sau, top_k=1),
            'micro/accuracy': torchmetrics.Accuracy(task="multiclass", **common_args_micro_sau),
            'micro/iou': torchmetrics.JaccardIndex(task="multiclass", **common_args_micro_sau),
            'micro/dice': torchmetrics.segmentation.DiceScore(**common_args_micro_sau),
            'micro/precision': torchmetrics.Precision(task='multiclass', **common_args_micro_sau, top_k=1),
            'micro/recall': torchmetrics.Recall(task='multiclass', **common_args_micro_sau, top_k=1),
            })
        
        self.metrics_lc = metrics_lc.clone(prefix='land_cover/')
        self.metrics_sau = metrics_sau.clone(prefix='soil_agricultural_use/')

        # test metrics
        self.metrics_lc_test = metrics_lc.clone(prefix='test/land_cover/')
        self.metrics_sau_test = metrics_sau.clone(prefix='test/soil_agricultural_use/')
        self.confusionMat_lc = torchmetrics.ConfusionMatrix(task='multiclass', ignore_index=0, num_classes=self.conf['n_classes_lc'], normalize='true')
        self.confusionMat_sau = torchmetrics.ConfusionMatrix(task='multiclass', ignore_index=0, num_classes=self.conf['n_classes_sau'], normalize='true')

        common_args_class_lc = {'average': 'none', 'num_classes': self.conf['n_classes_lc']}#, 'ignore_index':0, 'device':self.device}
        common_args_class_sau = {'average': 'none', 'num_classes': self.conf['n_classes_sau']}#, 'ignore_index':0, 'device':self.device}
        
        metrics_class_lc = MetricCollection({
            'class/accuracy': torchmetrics.Accuracy(task="multiclass", **common_args_class_lc),
            'class/iou': torchmetrics.JaccardIndex(task="multiclass", **common_args_class_lc),
            'class/precision': torchmetrics.Precision(task='multiclass', **common_args_class_lc, top_k=1),
            'class/recall': torchmetrics.Recall(task='multiclass', **common_args_class_lc, top_k=1)
            })

        metrics_class_sau = MetricCollection({
            'class/accuracy': torchmetrics.Accuracy(task="multiclass", **common_args_class_sau),
            'class/iou': torchmetrics.JaccardIndex(task="multiclass", **common_args_class_sau),
            'class/precision': torchmetrics.Precision(task='multiclass', **common_args_class_sau, top_k=1),
            'class/recall': torchmetrics.Recall(task='multiclass', **common_args_class_sau, top_k=1)
            })
        
        self.metrics_class_lc = metrics_class_lc.clone(prefix='land_cover/')
        self.metrics_class_sau = metrics_class_sau.clone(prefix='soil_agricultural_use/')

        self.model_name = self.conf['encoder_name']
        self.experiment_folder = ''

    def forward(self, batch):
        raise NotImplementedError('Base must be extended by child class that specifies the forward method.')


    def training_step(self, batch, batch_nb):

        inputs, targets, _ = batch # for the name of the image

        # append
        logits = self(inputs)
        gt_lc, gt_sau = targets
     
        # split logits
        logits_lc = logits[:,:self.conf['n_classes_lc']]
        logits_sau = logits[:,self.conf['n_classes_lc']:self.conf['n_classes_lc'] + self.conf['n_classes_sau']]
        # apply loss

        loss_lc = F.cross_entropy(logits_lc, gt_lc.squeeze(dim=1), weight=1-torch.tensor([1, 0.08169914,
                                                                                          0.05278166,
                                                                                          0.16236057,
                                                                                          0.02600136,
                                                                                          0.31404807,
                                                                                          0.33979217,
                                                                                          0.02331703], device=logits_lc.device), ignore_index=0)
        loss_sau = F.cross_entropy(logits_sau, gt_sau.squeeze(dim=1), weight=1-torch.tensor([1,
                                                                                          0.01157833,
                                                                                          0.06146465,
                                                                                          0.13093557,
                                                                                          0.0496246,
                                                                                          0.40127885,
                                                                                          0.01595872,
                                                                                          0.13955435,
                                                                                          0.01651926,
                                                                                          0.17308567], device=logits_lc.device), ignore_index=0)

        loss = self.conf['alpha']*loss_lc + (1-self.conf['alpha'])*loss_sau

        self.log("losses/land_cover", loss_lc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        self.log("losses/agricolture", loss_sau, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        self.log("losses/total", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])

        # show images
        if batch_nb % 500 == 0:
            _, class_lc = torch.max(logits_lc, 1)
            _, class_sau = torch.max(logits_sau, 1)
            self.show_images(class_lc, class_sau, inputs, gt_lc, gt_sau)

        return loss
    
    def validation_step(self, batch, batch_idx):

        inputs, targets, _ = batch

        # get input and semantic layout
        logits = self(inputs)
        gt_lc, gt_sau = targets

        # split logits
        logits_lc = logits[:,:self.conf['n_classes_lc']]
        logits_sau = logits[:,self.conf['n_classes_lc']:self.conf['n_classes_lc'] + self.conf['n_classes_sau']]

        # log validation loss
        loss_lc = F.cross_entropy(logits_lc, gt_lc.squeeze(dim=1), weight=1-torch.tensor([1-torch.finfo().eps,
                                                                                          0.08169914,
                                                                                          0.05278166,
                                                                                          0.16236057,
                                                                                          0.02600136,
                                                                                          0.31404807,
                                                                                          0.33979217,
                                                                                          0.02331703], device=logits_lc.device), ignore_index=0)        
        loss_sau = F.cross_entropy(logits_sau, gt_sau.squeeze(dim=1), weight=1-torch.tensor([1,
                                                                                          0.01157833,
                                                                                          0.06146465,
                                                                                          0.13093557,
                                                                                          0.0496246,
                                                                                          0.40127885,
                                                                                          0.01595872,
                                                                                          0.13955435,
                                                                                          0.01651926,
                                                                                          0.17308567], device=logits_lc.device), ignore_index=0)

        loss = self.conf['alpha']*loss_lc + (1-self.conf['alpha'])*loss_sau
        self.log("losses/val_lc", loss_lc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        self.log("losses/val_sau", loss_sau, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])
        self.log("losses/val_total", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.conf['batch_size'])

        # view images
        if batch_idx % 500 == 0:
            _, class_lc = torch.max(logits_lc, 1)
            _, class_sau = torch.max(logits_sau, 1)
            self.show_images(class_lc, class_sau, inputs, gt_lc, gt_sau, stage='val')

        self.metrics_lc.update(logits_lc, gt_lc)
        self.metrics_sau.update(logits_sau, gt_sau)

    def test_step(self, batch, batch_idx):

        self.experiment_folder = './torchmetrics_results/' + self.model_name + "_version_" + str(self.logger.version)

        if not(os.path.exists(self.experiment_folder)):
                os.mkdir(self.experiment_folder)

        inputs, targets, names = batch

        # get input and semantic layout
        logits = self(inputs)
        gt_lc, gt_sau = targets
        
        logits_lc = logits[:,:self.conf['n_classes_lc']]
        logits_sau = logits[:,self.conf['n_classes_lc']:self.conf['n_classes_lc'] + self.conf['n_classes_sau']]

        # view images
        if batch_idx == 0:
            _, class_lc = torch.max(logits_lc, 1)
            _, class_sau = torch.max(logits_sau, 1)
            self.show_images(class_lc, class_sau, inputs, gt_lc, gt_sau, stage='test')

        self.metrics_lc_test.update(logits_lc, gt_lc)
        self.metrics_sau_test.update(logits_sau, gt_sau)
        self.confusionMat_lc.update(logits_lc, gt_lc)
        self.confusionMat_sau.update(logits_sau, gt_sau)
        self.metrics_class_lc.update(logits_lc, gt_lc)
        self.metrics_class_sau.update(logits_sau, gt_sau)

        # save land_cover
        labels = ['Background','Building','Road','Residential','Industrial','Forest','Farmland','Water']
        self._show_and_save_predictions(inputs=inputs, logits=logits_lc, gt=gt_lc, num_classes=self.conf['n_classes_lc'], names=names, label=labels, folder='land_cover/', mode='land_cover')

        labels=[
            'Background',
            'Other agricultural crops',
            'Forage crops',
            'Corn',
            'Industrial plants',
            'Rice',
            'Seeds',
            'Man-made areas',
            'Water bodies',
            'Natural vegetation']

        self._show_and_save_predictions(inputs=inputs, logits=logits_sau, gt=gt_sau, num_classes=self.conf['n_classes_sau'], names=names, label=labels, folder='soil_agricultural_use/', mode='soil_agricultural_use')
   

    def on_validation_epoch_end(self):

        output_lc = self.metrics_lc.compute()
        output_sau = self.metrics_sau.compute()

        self.log_dict(output_lc)
        self.log_dict(output_sau)

        self.metrics_lc.reset()
        self.metrics_sau.reset()

    def on_test_epoch_end(self):
 
        results_path = self.experiment_folder

        # confusion matrix
        confMat_lc = self.confusionMat_lc.compute()
        confMat_sau = self.confusionMat_sau.compute()

        labels = ['Background',
                  'Building',
                  'Roads',
                  'Residential',
                  'Industrial',
                  'Forest',
                  'Farmland',
                  'Water']
        self._save_confusionMatrix(confMat_lc, results_path, labels, mode='land_cover')

        labels=[
            'Background',
            'Other agricultural crops',
            'Forage crops',
            'Corn',
            'Industrial plants',
            'Rice',
            'Seeds',
            'Man-made areas',
            'Water bodies',
            'Natural vegetation']
                    
        self._save_confusionMatrix(confMat_sau, results_path, labels, mode='soil_agricultural_use')

        self.confusionMat_lc.reset()
        self.confusionMat_sau.reset()

        # Metrics

        output_lc = self.metrics_lc_test.compute()
        output_sau = self.metrics_sau_test.compute()

        self.log_dict(output_lc)
        self.log_dict(output_sau)

        results_to_save = ''
        for cur_metric_name in output_lc:
            results_to_save += cur_metric_name + ' : ' + str(output_lc[cur_metric_name].item()) + '\n'

        self._save_list_results(results_path, "/test_metrics_lc.txt", (results_to_save,)) # list is to not create a different save_list function

        results_to_save = ''
        for cur_metric_name in output_sau:
            results_to_save += cur_metric_name + ' : ' + str(output_sau[cur_metric_name].item()) + '\n'

        self._save_list_results(results_path, "/test_metrics_sau.txt", (results_to_save,)) # list is to not create a different save_list function

        self.metrics_lc_test.reset()
        self.metrics_sau_test.reset()

        # per class
        output_lc = self.metrics_class_lc.compute()
        output_sau = self.metrics_class_sau.compute()

        for metrics in output_lc:
            results_to_save = ''
            computed_metrics = output_lc[metrics]
            for metric_value in computed_metrics:
                results_to_save += str(metric_value.item()) + '\n'

            self._save_list_results(results_path, "/land_cover_class_" + metrics.split('/')[-1] + ".txt", (results_to_save,)) # list is to not create a different save_list function

        for metrics in output_sau:
            results_to_save = ''
            computed_metrics = output_sau[metrics]
            for metric_value in computed_metrics:
                results_to_save += str(metric_value.item()) + '\n'

            self._save_list_results(results_path, "/soil_agricultural_use_class_" + metrics.split('/')[-1] + ".txt", (results_to_save,)) # list is to not create a different save_list function

        self.metrics_class_lc.reset()
        self.metrics_class_sau.reset()


    def _save_confusionMatrix(self, confMat, results_path, labels, mode='land_cover'):
        
        confMat = confMat.cpu().detach().numpy()
        
        # save confmat
        res_mat = ''
        for row_el in range(confMat.shape[0]):
            for col_el in range(confMat.shape[1]):
                res_mat += str(confMat[row_el, col_el].item()) +  " "
            res_mat += "\n"

        self._save_list_results(results_path, "/" + mode + "_confusion_matrix.txt", (res_mat,)) # list is to not create a different save_list function

        if mode == 'soil_agricultural_use':
            sns.set(font_scale=0.3)

        sns.heatmap(confMat, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.tight_layout()
        plt.savefig(results_path + '/' + mode + '_confusion_matrix.png', dpi=200)

        plt.clf()


    def _show_and_save_predictions(self, inputs, logits, gt, num_classes, names, folder, label, mode='land_cover'):

        ignore_index = 0
        im_mIoU = torchmetrics.JaccardIndex(task="multiclass", average='none', num_classes=num_classes, ignore_index=ignore_index)

        if not(self.conf['pca']):
            rgb, _, _, _, _ = inputs
        else:
            if mode == 'land_cover':
                rgb, _, _ = inputs
            else:
                rgb, hs, _ = inputs


        fontsize = 15
        for im_batch in range(logits.shape[0]):

            rgb2show = rearrange(rgb[im_batch], 'c h w -> h w c').cpu().float().numpy()
            rgb2show = (rgb2show -  np.min(rgb2show)) / (np.max(rgb2show) - np.min(rgb2show))
            
            if not(mode == 'land_cover'):
                hs2show = rearrange(hs[im_batch], 'c h w -> h w c').cpu().float().numpy()
                hs2show = hs2show[:,:,(50,39,30)] # bands to show from hs
                hs2show = (hs2show -  np.min(hs2show)) / (np.max(hs2show) - np.min(hs2show))

            out2show = logits[im_batch].argmax(dim=0).cpu().detach().float().numpy()/num_classes
            gt2show = gt[im_batch].squeeze().cpu().detach().float().numpy()/num_classes

            out2show[gt2show==0] = 0 # Masked

            color_map = 'Paired'
            if mode == 'land_cover':
                fig, (ax1, ax6, ax7) = plt.subplots(1,3, figsize=(20,10))
            else:
                fig, (ax1, ax6, ax7) = plt.subplots(1,3, figsize=(20,10))

            mask_log = logits[im_batch].argmax(dim=0)
            mask_log[gt[im_batch]==0] = 0
            im_mIoU.update(mask_log.unsqueeze(dim=0).cpu(), gt[im_batch].unsqueeze(dim=0).cpu())
            mIoU = im_mIoU.compute()
            # consider only present classes
            present_classes = torch.cat((mask_log.unique(), gt[im_batch].unique())).unique().cpu()
            present_classes = present_classes[present_classes!=ignore_index]
            mIoU = mIoU[present_classes].mean()
            fig.suptitle(str(mIoU))
            im_mIoU.reset()
            
            ax1.imshow(rgb2show)
            ax1.axis('off')
            ax1.set_title('RGB', fontsize=fontsize)
            ax6.imshow(gt2show, cmap=color_map, vmin=0, vmax=1)
            ax6.axis('off')
            ax6.set_title('GT', fontsize=fontsize)

            ax7.imshow(out2show, cmap=color_map, vmin=0, vmax=1)
            ax7.axis('off')
            ax7.set_title('Prediction', fontsize=fontsize)

            # legend
            cmap = plt.get_cmap('Paired')
            norm = plt.Normalize(vmin=0, vmax=len(label))

            handles = [plt.Rectangle((0, 0), 0, 0, color=cmap(norm(i)), label=label[i]) for i in range(len(label))]
            ax7.legend(handles=handles, fontsize=fontsize, loc=(1.04, 0.1))
            plt.tight_layout()

            if not(os.path.exists(self.experiment_folder + '/test_images/')):
                os.mkdir(self.experiment_folder + '/test_images/')

            if not(os.path.exists(self.experiment_folder + '/test_images/' + folder)):
                os.mkdir(self.experiment_folder + '/test_images/' + folder)

            fig.savefig(self.experiment_folder + '/test_images/' + folder + names[im_batch].split('.')[0] + '.png', dpi=200)

            fig.clear()
            plt.clf()
            plt.close(fig)


    # SAVE MODEL RESULTS
    def _save_list_results(self, results_path, name_list, data):
        if not(os.path.exists(results_path)):
            os.mkdir(results_path)
        
        with open(results_path + name_list, 'w') as fp:
            for item in data:
                fp.write(str(item) + "\n")    

    # show images
    def color_coding(self, semlay, n_classes, cmap='Paired'):
        # get colormap
        cmap = plt.get_cmap('Paired')
        # apply colormap
        rgb_out = cmap((semlay.cpu().float()/n_classes).numpy())
        # fix dims
        rgb_out = rearrange(torch.from_numpy(rgb_out), 'b h w c -> b c h w')
        # return
        return rgb_out

    def show_images(self, out_lc, out_sau, inputs, gt_lc, gt_sau, stage='train'):
        
        # move to cpu and detach
        rgb, hs, dem = inputs

        out_lc = out_lc.cpu().detach()
        out_sau = out_sau.cpu().detach()
        rgb = rgb.cpu().detach()
        hs = hs.cpu().detach()
        dem = dem.cpu().detach()
        gt_lc = gt_lc.cpu().detach()
        gt_sau = gt_sau.cpu().detach()
        # create masked outputs
        out_lc_masked = torch.zeros_like(out_lc)
        out_sau_masked = torch.zeros_like(out_sau)
        # for each image, remove content in background
        for i in range(out_lc.shape[0]):
            cur_lc_masked = out_lc[i].clone()
            cur_lc_gt = gt_lc[i]
            cur_lc_masked[cur_lc_gt==0] = 0
            out_lc_masked[i] = cur_lc_masked
            cur_sau_masked = out_sau[i].clone()
            cur_sau_gt = gt_sau[i]
            cur_sau_masked[cur_sau_gt==0] = 0
            out_sau_masked[i] = cur_sau_masked
        # color coding
        out_lc = self.color_coding(out_lc, self.conf['n_classes_lc'])
        out_sau = self.color_coding(out_sau, self.conf['n_classes_sau'])
        out_lc_masked = self.color_coding(out_lc_masked, self.conf['n_classes_lc'])
        out_sau_masked = self.color_coding(out_sau_masked, self.conf['n_classes_sau'])
        gt_lc = self.color_coding(gt_lc, self.conf['n_classes_lc'])
        gt_sau = self.color_coding(gt_sau, self.conf['n_classes_sau'])
        # normalize other images
        dem = (dem - dem.min()) / (dem.max() - dem.min())
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        hs = (hs - hs.min()) / (hs.max() - hs.min())
        # sanitize images with nan
        out_lc_masked = torch.nan_to_num(out_lc_masked, nan=0.0)
        out_sau_masked = torch.nan_to_num(out_sau_masked, nan=0.0)
        out_lc = torch.nan_to_num(out_lc, nan=0.0)
        out_sau = torch.nan_to_num(out_sau, nan=0.0)
        # clamp
        out_lc_masked = torch.clamp(out_lc_masked, min=0, max=1)
        out_sau_masked = torch.clamp(out_sau_masked, min=0, max=1)
        out_lc = torch.clamp(out_lc, min=0, max=1)
        out_sau = torch.clamp(out_sau, min=0, max=1)
        rgb = torch.clamp(rgb, min=0, max=1)
        hs = torch.clamp(hs, min=0, max=1)
        dem = torch.clamp(dem, min=0, max=1)
        # display semantic layouts
        self.logger.experiment.add_image(f'{stage}/land_cover/out_masked', utils.make_grid(out_lc_masked), self.global_step)
        self.logger.experiment.add_image(f'{stage}/soil_agricultural_use/out_masked', utils.make_grid(out_sau_masked), self.global_step)
        self.logger.experiment.add_image(f'{stage}/land_cover/out', utils.make_grid(out_lc), self.global_step)
        self.logger.experiment.add_image(f'{stage}/soil_agricultural_use/out', utils.make_grid(out_sau), self.global_step)
        self.logger.experiment.add_image(f'{stage}/land_cover/gt', utils.make_grid(gt_lc), self.global_step)
        self.logger.experiment.add_image(f'{stage}/soil_agricultural_use/gt', utils.make_grid(gt_sau), self.global_step)
        # display input images
        self.logger.experiment.add_image(f'{stage}/inputs/rgb', utils.make_grid(rgb), self.global_step)
        self.logger.experiment.add_image(f'{stage}/inputs/hs', utils.make_grid(hs.mean(1).unsqueeze(1)), self.global_step)
        self.logger.experiment.add_image(f'{stage}/inputs/dem', utils.make_grid(dem), self.global_step)

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

        LR_scheduler = {"scheduler": StepLR(optimizer, step_size=30, gamma=0.85), "monitor": "losses/val_total",
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

            if conf['alpha'] == 0:
                callbacks = [
                        RichProgressBar(),
                        ModelCheckpoint(monitor='soil_agricultural_use/macro/accuracy', mode='max', save_top_k=1, save_last=True, filename='soil_agricultural_use/macro_accuracy'), # True
                        ModelCheckpoint(monitor='soil_agricultural_use/macro/iou', mode='max', save_top_k=1, save_last=False, filename='soil_agricultural_use/macro_iou'),
                        ModelCheckpoint(monitor='soil_agricultural_use/micro/accuracy', mode='max', save_top_k=1, save_last=True, filename='soil_agricultural_use/micro_accuracy'), # True
                        ModelCheckpoint(monitor='soil_agricultural_use/micro/iou', mode='max', save_top_k=1, save_last=False, filename='soil_agricultural_use/micro_iou'),
                        LearningRateMonitor(logging_interval='epoch'),
                        # EarlyStopping(monitor="val_loss", mode="min", patience=20)
                    ]
            elif conf['alpha'] == 1:
                callbacks = [
                        RichProgressBar(),
                        ModelCheckpoint(monitor='land_cover/macro/accuracy', mode='max', save_top_k=1, save_last=False, filename='land_cover/macro_accuracy'),
                        ModelCheckpoint(monitor='land_cover/macro/iou', mode='max', save_top_k=1, save_last=True, filename='land_cover/macro_iou'),
                        ModelCheckpoint(monitor='land_cover/micro/accuracy', mode='max', save_top_k=1, save_last=False, filename='land_cover/micro_accuracy'),
                        ModelCheckpoint(monitor='land_cover/micro/iou', mode='max', save_top_k=1, save_last=True, filename='land_cover/micro_iou'),
                        LearningRateMonitor(logging_interval='epoch'),
                    ]
            else:
                callbacks = [
                        RichProgressBar(),
                        ModelCheckpoint(monitor='land_cover/macro/accuracy', mode='max', save_top_k=1, save_last=False, filename='land_cover/macro_accuracy'),
                        ModelCheckpoint(monitor='land_cover/macro/iou', mode='max', save_top_k=1, save_last=True, filename='land_cover/macro_iou'),
                        ModelCheckpoint(monitor='land_cover/micro/accuracy', mode='max', save_top_k=1, save_last=False, filename='land_cover/micro_accuracy'),
                        ModelCheckpoint(monitor='land_cover/micro/iou', mode='max', save_top_k=1, save_last=True, filename='land_cover/micro_iou'),
                        ModelCheckpoint(monitor='soil_agricultural_use/macro/accuracy', mode='max', save_top_k=1, save_last=True, filename='soil_agricultural_use/macro_accuracy'), # True
                        ModelCheckpoint(monitor='soil_agricultural_use/macro/iou', mode='max', save_top_k=1, save_last=False, filename='soil_agricultural_use/macro_iou'),
                        ModelCheckpoint(monitor='soil_agricultural_use/micro/accuracy', mode='max', save_top_k=1, save_last=True, filename='soil_agricultural_use/micro_accuracy'), # True
                        ModelCheckpoint(monitor='soil_agricultural_use/micro/iou', mode='max', save_top_k=1, save_last=False, filename='soil_agricultural_use/micro_iou'),
                        LearningRateMonitor(logging_interval='epoch'),
                    ]
            #Early Stopping
            early_stop_callback = EarlyStopping(
            monitor='val_loss',        # Metric to observe
            min_delta=0.00,            # minimmum change to consider an improvement
            patience=4,                # nbre of epoch without improvement before stopping
            verbose=True,              # print messages of early stopping
            mode='min',                # minimizing the metric
            strict=True,               # if True, observed metric must be present at each epoch
    )
           # define trainer
            trainer = pl.Trainer(
                            accelerator='gpu',
                            devices=1,
                            max_epochs=conf['n_epochs'] if 'n_epochs' in conf else 1000,
                            # max_epochs=250,
                            num_sanity_val_steps = 2 if args.checkpoint == '' else 0,
                            logger=TensorBoardLogger('checkpoints', name=conf['experiment_name']),
                            profiler="simple",
                            callbacks=[*callbacks,early_stop_callback],
                            deterministic='warn')
            if args.checkpoint == '':
                trainer.fit(model)
            else:
                trainer.fit(model,ckpt_path=args.checkpoint)

            # test
            trainer.test(model)
        else:

            #test
            model = cur_class.load_from_checkpoint(args.checkpoint)
            conf = model.conf
            
            tester = pl.Trainer()
            tester.test(model)
            
