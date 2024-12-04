import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import model.loss as module_loss
import wandb

selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, samples_per_cls=None, no_of_classes=5, beta=0.9999, gamma=2.0):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # adjust log step if more logs are needed

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0

        # Parameters for Class-Balanced Loss
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :param total_epochs: Total number of epochs.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            # # Apply Class-Balanced Loss if specified
            # if self.config['loss']['type'] == "CB_loss":
            #     loss = module_loss.CB_loss(
            #         labels=target,
            #         logits=output,
            #         samples_per_cls=self.samples_per_cls,
            #         no_of_classes=self.no_of_classes,
            #         loss_type="focal",
            #         beta=self.beta,
            #         gamma=self.gamma
            #     )
            # else:
            #     # If not using CB_loss, use the standard criterion
            #     loss = self.criterion(output, target, self.class_weights, self.device)
            loss = self.criterion(output, target, self.class_weights, self.device)

            loss.backward()
            self.optimizer.step()

            # Update metrics
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
            wandb.log({"train_loss": loss.item()})

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        wandb.log(log)

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # Reduce learning rate after epoch 10
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])

            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # # Apply Class-Balanced Loss if specified
                # if self.config['loss']['type'] == "CB_loss":
                #     loss = module_loss.CB_loss(
                #         labels=target,
                #         logits=output,
                #         samples_per_cls=self.samples_per_cls,
                #         no_of_classes=self.no_of_classes,
                #         loss_type="focal",
                #         beta=self.beta,
                #         gamma=self.gamma
                #     )
                # else:
                #     loss = self.criterion(output, target, self.class_weights, self.device)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    metric_value = met(output, target)
                    self.valid_metrics.update(met.__name__, metric_value)
                    wandb.log({f"val_{met.__name__}": metric_value})

                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())

        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
