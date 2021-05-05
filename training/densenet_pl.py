#!/usr/bin/env python3

# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.metrics import MeanAbsoluteError


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(pl.LightningModule):
    def __init__(self,
                 num_input_features,
                 growth_rate,
                 bn_size,
                 drop_rate,
                 efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad
                                  for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseBlock(pl.LightningModule):
    def __init__(self,
                 num_layers,
                 num_input_features,
                 bn_size,
                 growth_rate,
                 drop_rate,
                 efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(pl.LightningModule):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self,
                 growth_rate=12,
                 block_config=(32, 32),
                 compression=0.5,
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 chan=1,
                 num_classes=1,
                 small_inputs=True,
                 efficient=False):

        super(DenseNet, self).__init__()
        self.save_hyperparameters()
        self.mean_absolute_error = pl.metrics.MeanAbsoluteError()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv0',
                     nn.Conv3d(chan,
                               num_init_features,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)),
                ]))
        else:
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv0',
                     nn.Conv3d(chan,
                               num_init_features,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)),
                ]))
            self.features.add_module('norm0',
                                     nn.BatchNorm3d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module(
                'pool0',
                nn.MaxPool3d(kernel_size=3,
                             stride=2,
                             padding=1,
                             ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features *
                                                            compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def mae_loss(self, logits, labels):
        return F.l1_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.mae_loss(logits, y)
        mae = self.mean_absolute_error(logits, y)
        self.log('train_loss',
                 loss,
                 sync_dist=True,
                 on_epoch=True,
                 on_step=True)
        self.log('train_mae', mae, sync_dist=True, on_epoch=True, on_step=True)
        self.log('epoch_num',
                 self.current_epoch,
                 sync_dist=True,
                 on_epoch=True,
                 on_step=True)
        return {'loss': loss, 'mae': mae}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['mae'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, on_epoch=True, sync_dist=True)
        self.log('avg_train_mae', avg_mae, on_epoch=True, sync_dist=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.mae_loss(logits, y)
        mae = self.mean_absolute_error(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_mae', mae, on_step=True, on_epoch=True, sync_dist=True)
        return {'rval_loss': loss, 'rval_mae': mae}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['rval_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['rval_mae'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, sync_dist=True)
        self.log('avg_val_mae', avg_mae, on_epoch=True, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.mae_loss(logits, y)
        mae = self.mean_absolute_error(logits, y)
        self.log('test_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        self.log('test_mae', mae, on_step=True, on_epoch=True, sync_dist=True)
        return {
            'rtest_loss': loss,
            'rtest_mae': mae,
            'logits': logits,
            'y_vals': y
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['rtest_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['rtest_mae'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_epoch=True, sync_dist=True)
        self.log('avg_test_mae', avg_mae, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=0.1,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=1e-4)

        lr_scheduler = {
            'scheduler':
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.75,
                                                       patience=5,
                                                       threshold=5e-3,
                                                       threshold_mode='abs',
                                                       cooldown=0,
                                                       min_lr=1e-6,
                                                       verbose=True),
            'name':
            'red_pl_lr',
            'monitor':
            'train_loss_epoch'
        }

        return [optimizer], [lr_scheduler]
