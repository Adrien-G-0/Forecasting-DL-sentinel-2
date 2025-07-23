"""
Model Training and Evaluation Script

This script provides functionality for training and evaluating neural network models
for multi-source data fusion. It supports both CNN-based and Transformer-based architectures.

Key Features:
- Trains CNN models from NewArchitectures module
- Loads and tests models from checkpoints
- Trains Transformer models from TransformerArchitectures module
- Supports multiple data sources (esa, sau, lc, etc.)
- Uses JSON configuration files for flexible architecture setup
- Implements model checkpointing, early stopping, and TensorBoard logging

Usage:
Run this script directly to execute the training pipelines. Modify the configuration
files to experiment with different model architectures and hyperparameters.
"""

from NewArchitectures import NewArchitectures
from TransformerArchitectures import TransformerArchitectures
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import json

# Define different combinations of data sources to experiment with
diff_sources = [
    ['esa', 'sau', 'lc']
]

# Define fusion methods to experiment with
fusion_type = ["pixel_to_pixel"]

if __name__ == '__main__':
    # Training pipeline for CNN models
    if True:
        for sources in diff_sources:
            for fusion in fusion_type:
                # Load and update configuration
                with open('params.json') as f:
                    conf = json.load(f)
                conf['sources'] = sources
                conf['method'] = fusion

                # Save updated configuration
                with open('params.json', 'w') as f:
                    json.dump(conf, f, indent=4)

                # Create model and training setup
                model = NewArchitectures(conf)

                # Define callbacks for training
                callbacks = [
                    RichProgressBar(),
                    ModelCheckpoint(
                        monitor='val_mae',
                        mode='min',
                        save_top_k=1,
                        save_last=True,
                        filename='l1-{epoch:02d}-{val_loss:.3f}'
                    ),
                    LearningRateMonitor(logging_interval='epoch'),
                    EarlyStopping(
                        monitor='val_mae',
                        min_delta=0.00,
                        patience=20,
                        verbose=True,
                        mode='min',
                        strict=True,
                    )
                ]

                # Configure and run trainer
                trainer = pl.Trainer(
                    accelerator='gpu',
                    devices=1,
                    max_epochs=5,
                    num_sanity_val_steps=2,
                    logger=TensorBoardLogger('checkpoints', name='test/'+conf['experiment_name'] + "_".join(conf['sources']+[conf['method']])),
                    callbacks=callbacks
                )

                # Train and test model
                print("Training...")
                print(conf['sources'])
                print(conf['method'])
                trainer.fit(model)
                print("Testing best model...")
                trainer.test(ckpt_path="best")

    # Load and test models from checkpoints
    if False:
        ckpt_path = 'checkpoints/17_07/lc_late_fusion/version_0/checkpoints/l1-epoch=36-val_loss=0.220.ckpt'
        # model = TransformerArchitectures.load_from_checkpoint(ckpt_path)
        model = NewArchitectures.load_from_checkpoint(ckpt_path)
        conf = model.conf

        # Setup training configuration
        callbacks = [
            RichProgressBar(),
            ModelCheckpoint(
                monitor='val_mae',
                mode='min',
                save_top_k=1,
                save_last=True,
                filename='l1_loss-{epoch:02d}-{val_loss:.3f}'
            ),
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(
                monitor='val_mae',
                min_delta=0.00,
                patience=20,
                verbose=True,
                mode='min',
                strict=True,
            )
        ]

        # Configure trainer with GPU acceleration
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=10,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger('checkpoints', name=conf['experiment_name'] + "_".join(conf['sources']+[conf['method']])),
            callbacks=callbacks
        )

        # Continue training or test model
        trainer.fit(model, ckpt_path=ckpt_path)
        trainer.test(ckpt_path="best")

    # Training pipeline for Transformer models
    if False:
        for sources in diff_sources:
            # Load and update Transformer configuration
            with open('Tparams.json', 'r') as f:
                conf = json.load(f)
            conf['sources'] = sources

            # Save updated configuration
            with open('Tparams.json', 'w') as f:
                json.dump(conf, f, indent=4)

            # Create Transformer model
            model = TransformerArchitectures(conf)

            # Define callbacks for training
            callbacks = [
                RichProgressBar(),
                ModelCheckpoint(
                    monitor='val_mae',
                    mode='min',
                    save_top_k=1,
                    save_last=True,
                    filename='l1_loss-{epoch:02d}-{val_loss:.3f}'
                ),
                LearningRateMonitor(logging_interval='epoch'),
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.001,
                    patience=30,
                    verbose=True,
                    mode='min',
                    strict=True,
                )
            ]

            # Configure trainer with GPU acceleration
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=1,
                max_epochs=conf['n_epochs'],
                num_sanity_val_steps=2,
                logger=TensorBoardLogger('checkpoints', name=conf['experiment_name'] + "_".join(conf['sources'] + [conf['method']])),
                callbacks=callbacks
            )

            # Train and test Transformer model
            print("Training...")
            print(conf['sources'])
            print(conf['method'])
            trainer.fit(model)
            trainer.test(ckpt_path="best")
