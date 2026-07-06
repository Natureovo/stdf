import torch
import torch.nn as nn
import torch.nn.functional as F

from degradation_features import summarize_budget_features


class BudgetNet(nn.Module):
    """Predict frame-level local generation budget from no-reference features."""

    def __init__(
            self,
            in_dim=18,
            hidden_dim=64,
            min_budget=0.02,
            max_budget=0.45):
        super(BudgetNet, self).__init__()
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lq, base, guidance=None, rate_cond=None):
        features = summarize_budget_features(
            lq,
            base,
            guidance=guidance,
            rate_cond=rate_cond,
        )
        raw = torch.sigmoid(self.mlp(features))
        budget = self.min_budget + (self.max_budget - self.min_budget) * raw
        return budget.clamp(self.min_budget, self.max_budget)


def oracle_budget_from_guidance(
        oracle_guidance,
        threshold=0.20,
        target_mode='mean_guidance'):
    guidance = oracle_guidance.detach().clamp(0, 1)
    if target_mode in ('mean_guidance', 'soft_mean', 'continuous_mean'):
        return guidance.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
    if target_mode in ('threshold_coverage', 'hard_coverage'):
        oracle_mask = guidance >= threshold
        return oracle_mask.float().mean(dim=(1, 2, 3), keepdim=False).view(-1, 1)
    raise ValueError(f'Unsupported budget target_mode: {target_mode}')


def budget_prediction_losses(
        pred_budget,
        oracle_guidance,
        threshold=0.20,
        target_mode='mean_guidance',
        l1_weight=1.0,
        mse_weight=0.25):
    target_budget = oracle_budget_from_guidance(
        oracle_guidance,
        threshold=threshold,
        target_mode=target_mode,
    ).to(pred_budget.device)
    l1_loss = F.l1_loss(pred_budget, target_budget)
    mse_loss = F.mse_loss(pred_budget, target_budget)
    loss = l1_weight * l1_loss + mse_weight * mse_loss
    return {
        'loss': loss,
        'l1_loss': l1_loss,
        'mse_loss': mse_loss,
        'target_budget': target_budget,
    }


def build_budget_net(opts=None):
    opts = opts or {}
    return BudgetNet(
        in_dim=opts.get('in_dim', 18),
        hidden_dim=opts.get('hidden_dim', 64),
        min_budget=opts.get('min_budget', 0.02),
        max_budget=opts.get('max_budget', 0.45),
    )
