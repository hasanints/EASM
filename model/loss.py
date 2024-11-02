# import torch
# import torch.nn as nn


# def weighted_CrossEntropyLoss(output, target, classes_weights, device):
#     cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
#     return cr(output, target)

import numpy as np
import torch
import torch.nn.functional as F

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
    modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))) if gamma != 0.0 else 1.0
    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss) / torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type="focal", beta=0.9999, gamma=2.0):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    weights = torch.tensor(weights).float().unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1).unsqueeze(1).repeat(1, no_of_classes)

    if loss_type == "focal":
        return focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        return F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        return F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
