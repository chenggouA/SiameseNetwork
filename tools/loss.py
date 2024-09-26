import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
from torch import Tensor

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6 ):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # 确保 inputs 是 float 类型，并使用 softmax 来处理多分类任务
        inputs = torch.softmax(inputs, dim=1)
        
        # 将 targets 转换为 float 类型
        targets = targets.float()

        # 计算每个类别的 Dice Coefficient
        dice_loss = 0
        num_present_classes = 0  # 记录出现的类别数
        
        for i in range(self.num_classes):
            # 获取当前类别的预测概率和真实标签
            input_i = inputs[:, i, :, :]
            target_i = targets[:, i, :, :]

            # 计算 Dice Coefficient
            intersection = torch.sum(input_i * target_i)
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            
            if torch.sum(target_i) > 0:
                # 只有当目标类别在批次中出现时才计算 Dice Loss
                dice_loss += 1 - dice
                num_present_classes += 1
        
        if num_present_classes > 0:
            # 平均所有出现类别的 Dice Loss
            return dice_loss / num_present_classes
        else:
            # 如果没有类别出现，返回默认损失值（如 0）
            return torch.tensor(0.0, device=inputs.device)


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.CE_Loss =  nn.CrossEntropyLoss(ignore_index=num_classes, reduce='none')

    def forward(self, inputs: Tensor, targets: Tensor):

        c = inputs.shape[1]

        temp_inputs =  inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_targets = targets.contiguous().view(-1)
        # 计算交叉熵损失
        logpt = self.CE_Loss(temp_inputs, temp_targets)
        pt = torch.exp(-logpt)  # 得到预测概率

        # 计算 Focal Loss
        focal_loss = (1 - pt) ** self.gamma * logpt

        # 如果指定了 alpha，进行类别加权
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor([self.alpha, 1 - self.alpha])
            else:
                alpha = self.alpha
            at = alpha.gather(0, targets)
            focal_loss = focal_loss * at

        # 根据 reduction 参数聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # 不进行聚合，返回每个样本的损失
