import argparse
import collections
import numpy as np

from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *

import torch
import torch.nn as nn

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init_normal)
    logger.info(model)

    # Ridge Regularization parameter
    lambda_ridge = config['loss_args'].get('lambda_ridge', 100)  # Default value if not specified

    # Ensure 'num_classes' is in 'arch' args
    if 'args' not in config['arch'] or 'num_classes' not in config['arch']['args']:
        raise KeyError("`num_classes` must be defined in `arch` args in the configuration file.")

    # get function handles of loss and metrics
    if config['loss'] == 'CB_loss':
        # Ensure that necessary parameters are provided in the config
        samples_per_cls = config['loss_args'].get('samples_per_cls', [1]*config['arch']['args']['num_classes'])
        no_of_classes = config['arch']['args']['num_classes']
        beta = config['loss_args'].get('beta', 0.9999)  # Default beta value
        gamma = config['loss_args'].get('gamma', 2.0)  # Default gamma value
        loss_type = config['loss_args'].get('loss_type', 'focal')  # Default loss type

        # Define criterion as a lambda function to pass additional parameters to CB_loss
        criterion = lambda outputs, labels: module_loss.CB_loss(
            labels, outputs, samples_per_cls, no_of_classes, loss_type, beta, gamma
        ) + (lambda_ridge * sum(torch.norm(p, 2) for p in model.parameters()))  # Add Ridge penalty
    else:
        # Default loss from config (e.g., CrossEntropy)
        criterion = lambda outputs, labels: getattr(module_loss, config['loss'])(outputs, labels) + \
                                            (lambda_ridge * sum(torch.norm(p, 2) for p in model.parameters()))

    # Get metrics function handles
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # Data preparation and training
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



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
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

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)
