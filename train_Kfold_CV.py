# main.py

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
from metric_calculator import MetricsCalculator  # Updated import


import torch
import torch.nn as nn

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def weights_init_normal(m):
    """
    Initialize weights for the model layers using a normal distribution.
    """
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def main(config, fold_id):
    """
    Main function to set up data loaders, model, loss, optimizer, and start training.
    """
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init_normal)
    logger.info(model)

    # Debugging: Print available keys in config
    print("Available config keys:", config.config.keys())  # <-- Add this line

    # get function handles of loss and metrics
    loss_function = getattr(module_loss, config['loss'])  # Use CB_loss

    # Ensure 'loss_args' is available in config
    loss_args = config['loss_args'] if 'loss_args' in config.config else {}  # Access loss-specific arguments

    # Modify criterion to include all necessary loss arguments
    def criterion(output, target, class_weights, device):
        return loss_function(target, output, class_weights, 
                             config['arch']['args']['num_classes'],
                             loss_args['loss_type'], loss_args['beta'], loss_args['gamma'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # Load data for the specified fold
    data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], batch_size)
    weights_for_each_class = calc_class_weight(data_count)

    # Initialize Trainer and train the model
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      class_weights=weights_for_each_class)

    trainer.train()

    # Initialize MetricsCalculator after training is done
    metrics_calculator = MetricsCalculator(config, trainer.checkpoint_dir)
    # Calculate and save metrics
    metrics_calculator._calc_metrics()





if __name__ == '__main__':
    # Argument parsing
    args = argparse.ArgumentParser(description='PyTorch Training')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='Config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='Path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='Indices of GPUs to enable (default: 0)')
    args.add_argument('-f', '--fold_id', type=str,
                      help='Fold ID for cross-validation')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files for training')

    # Additional custom arguments
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    # Parse arguments
    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    # Configure GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args2.device

    # Load configuration from JSON file
    config = ConfigParser.from_args(args, fold_id, options)

    # Load data based on the provided directory
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    # Start main training process
    main(config, fold_id)
