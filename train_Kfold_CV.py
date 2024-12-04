import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import wandb

from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *


# Set random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def weights_init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main(config, fold_id):

    # # Inisialisasi W&B
    # wandb.init(
    #     project="nama_proyek_wandb",  # ganti dengan nama proyek yang diinginkan
    #     config=config._config,        # log seluruh konfigurasi
    #     name=f"Fold_{fold_id}"        # beri nama run per fold
    # )

    # # Log konfigurasi spesifik tambahan
    # wandb.config.update({
    #     "batch_size": config["data_loader"]["args"]["batch_size"],
    #     "epochs": config["trainer"]["epochs"],
    #     "fold_id": fold_id,
    # })

    batch_size = config["data_loader"]["args"]["batch_size"]
    logger = config.get_logger('train')

    # Build model architecture, initialize weights, and log model info
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init_normal)
    logger.info(model)

    # # Setup data loaders for current fold
    # data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
    #                                                                folds_data[fold_id][1], 
    #                                                                batch_size)

    # # Calculate class distribution (samples_per_cls) for CB_loss
    # samples_per_cls = data_count
    # no_of_classes = len(samples_per_cls)
    # beta = config['loss']['args'].get('beta', 0.9999)
    # gamma = config['loss']['args'].get('gamma', 2.0)
    # loss_type = config['loss']['args'].get('type', "focal")

    # # Set criterion (CB_loss or standard CrossEntropyLoss)
    # if config['loss']['type'] == 'CB_loss':
    #     criterion = lambda output, target: module_loss.CB_loss(
    #         labels=target,
    #         logits=output,
    #         samples_per_cls=samples_per_cls,
    #         no_of_classes=no_of_classes,
    #         loss_type=loss_type,
    #         beta=beta,
    #         gamma=gamma
    #     )
    # else:
    #     criterion = getattr(module_loss, config['loss']['type'])

    # # Get metrics as defined in config
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    # # Build optimizer
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # # Initialize Trainer with CB_loss parameters
    # trainer = Trainer(
    #     model=model,
    #     criterion=criterion,
    #     metric_ftns=metrics,
    #     optimizer=optimizer,
    #     config=config,
    #     data_loader=data_loader,
    #     fold_id=fold_id,
    #     valid_data_loader=valid_data_loader,
    #     samples_per_cls=samples_per_cls,
    #     no_of_classes=no_of_classes,
    #     beta=beta,
    #     gamma=gamma
    # )

    # # Start training
    # trainer.train()

     # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], batch_size)
    weights_for_each_class = calc_class_weight(data_count)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      class_weights=weights_for_each_class)

    trainer.train()
    
    # Akhiri logging W&B setelah selesai
    # wandb.finish()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Training with K-fold Cross-Validation')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    # Load config and folds data
    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)
