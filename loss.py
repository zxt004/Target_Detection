# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# 继续MultiModalLoss类
class MultiModalLoss(nn.Module):
    """多模态检测损失函数（完整RL和一致性实现）"""

    def __init__(self, config):
        super(MultiModalLoss, self).__init__()
        self.config = config

        # 检测损失
        self.class_loss = FocalLoss(alpha=config.get('focal_alpha', 0.25), gamma=config.get('focal_gamma', 2.0))
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')

        # 对齐损失
        self.align_loss = nn.CosineEmbeddingLoss()

        # 提示优化损失（RL策略梯度）
        self.rl_loss_weight = config.get('rl_loss_weight', 0.1)


    def forward(self, outputs, targets, rl_rewards=None):
        """计算完整的多任务损失"""
        # 1. 检测损失
        det_loss = self._compute_detection_loss(outputs, targets)

        # 2. 对齐损失
        align_loss = self._compute_alignment_loss(outputs, targets)

        # 3. 一致性损失
        consistency_loss = outputs['consistency_outputs']['consistency_loss']

        # 4. RL提示优化损失
        prompt_loss = torch.tensor(0.0, device=det_loss.device)
        if rl_rewards is not None and 'prompt_outputs' in outputs:
            prompt_loss = self._compute_prompt_rl_loss(outputs['prompt_outputs'], rl_rewards)

        # 总损失
        total_loss = (
                self.config['det_weight'] * det_loss +
                self.config['align_weight'] * align_loss +
                self.config['consistency_weight'] * consistency_loss +
                self.rl_loss_weight * prompt_loss
        )

        return {
            'total_loss': total_loss,
            'det_loss': det_loss,
            'align_loss': align_loss,
            'consistency_loss': consistency_loss,
            'prompt_loss': prompt_loss
        }

    def _compute_prompt_rl_loss(self, prompt_outputs, rewards):
        """计算RL策略损失"""
        # 仅在有多个样本时计算
        if prompt_outputs['action'].dim() == 0 or len(prompt_outputs['action']) < 2:
            return torch.tensor(0.0, device=rewards.device)

        # 标准化奖励
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 策略梯度损失
        policy_loss = -torch.sum(prompt_outputs['action_log_prob'] * normalized_rewards)

        # 价值损失
        value_loss = F.mse_loss(prompt_outputs['state_value'], rewards.unsqueeze(1))

        # 熵正则化
        entropy = -torch.sum(prompt_outputs['action_probs'] * torch.log(prompt_outputs['action_probs'] + 1e-8))
        entropy_bonus = 0.01 * entropy

        total_rl_loss = policy_loss + 0.5 * value_loss - entropy_bonus

        return total_rl_loss


class FocalLoss(nn.Module):
    """Focal Loss 用于类别不平衡问题"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss