#!/usr/bin/env python3

import argparse
import json
import copy
import time
import utils
import densenet_pl as dn

import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchsummary import summary
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='3D DenseNet Training')
parser.add_argument(
    '--data_path',
    default="/N/project/ankur_projects/gdb9/data/",
    type=str,
    help='directory path where training and test data are stored')
parser.add_argument('--channel',
                    type=int,
                    default=2,
                    help='channel number: 0,1,2,3')
parser.add_argument('--grid_length',
                    default=14,
                    type=int,
                    help='voxel grid length of the cubic volume')
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='batch size for one GPU')
parser.add_argument('--epochs',
                    default=250,
                    type=int,
                    help='total number of epochs')
parser.add_argument('--dense1',
                    default=16,
                    type=int,
                    help='depth of the first dense block')
parser.add_argument('--dense2',
                    default=16,
                    type=int,
                    help='depth of the second dense block')
args = parser.parse_args()


# Load data
class DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 ch=2,
                 k=14,
                 path="/N/project/ankur_projects/gdb9/data/"):
        super().__init__()
        self.batch_size = batch_size
        self.ch = ch
        self.k = k
        self.path = path

    def setup(self, stage=None):
        X_train, X_test, y_train, y_test = utils.make_data(ch=self.ch,
                                                           path=self.path,
                                                           k=self.k)

        X_train_tensor = torch.Tensor(X_train)
        X_test_tensor = torch.Tensor(X_test)
        y_train_tensor = torch.Tensor(y_train)
        y_train_tensor = y_train_tensor.unsqueeze(1)
        y_test_tensor = torch.Tensor(y_test)
        y_test_tensor = y_test_tensor.unsqueeze(1)

        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=True)


# Print results to standard output
class my_callbacks(Callback):
    def __init__(self) -> None:
        self.metrics: List = []

    def on_epoch_start(self, trainer: Trainer,
                       pl_module: LightningModule) -> None:
        self.tperf = time.perf_counter()
        self.tproc = time.process_time()

    def on_epoch_end(self, trainer: Trainer,
                     pl_module: LightningModule) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        new_metrics_dict = {k: v.item() for k, v in metrics_dict.items()}
        lr = trainer.optimizers[0].param_groups[0].get('lr')
        epoch = new_metrics_dict.get("epoch_num")
        loss = new_metrics_dict.get("train_loss_epoch")
        mae = new_metrics_dict.get("train_mae_epoch")
        val_mae = new_metrics_dict.get("val_mae_epoch")
        val_loss = new_metrics_dict.get("val_loss_epoch")
        #pl_module.print(json.dumps(new_metrics_dict, indent=4, sort_keys=True),
        #                flush=True)
        etperf = time.perf_counter() - self.tperf
        etproc = time.process_time() - self.tproc
        pl_module.print(
            "epoch: {:f} - lr: {:f} - perf_time: {:f}s - loss: {:f} - mae: {:f} - val_loss: {:f} - val_mae: {:f}"
            .format(epoch, lr, etperf, loss, mae, val_loss, val_mae),
            flush=True)


seed_everything(22)
p_callback = my_callbacks()
data_module = DataModule(batch_size=args.batch_size,
                         ch=args.channel,
                         k=args.grid_length,
                         path=args.data_path)

# create model
model = dn.DenseNet(growth_rate=12,
                    block_config=(args.dense1, args.dense2),
                    compression=0.5,
                    num_init_features=64,
                    bn_size=4,
                    drop_rate=0,
                    chan=1,
                    num_classes=1,
                    small_inputs=True,
                    efficient=False)

# Number of model parameters
#print('Number of model parameters: {}'.format(
#sum([p.data.nelement() for p in model.parameters()])))
#summary(model, (3, k, k, k), device='cpu')

# Save checkpoint files
checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch',
                                      verbose=True,
                                      save_last=True,
                                      save_top_k=2,
                                      mode='min')

# Train model
trainer = pl.Trainer(gpus=-1,
                     accelerator='ddp',
                     benchmark=True,
                     callbacks=[p_callback],
                     checkpoint_callback=checkpoint_callback,
                     max_epochs=args.epochs,
                     progress_bar_refresh_rate=0)

trainer.fit(model, data_module)
