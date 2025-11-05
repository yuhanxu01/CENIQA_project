"""Loss functions for CENIQA training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CENIQALoss(nn.Module):
    """Combined loss for CENIQA training."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.1)
        
    def contrastive_loss(self, features, labels):
        """Local-global contrastive loss."""
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T)
        
        # Positive pair mask
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Contrastive loss
        pos_sim = (sim_matrix * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        neg_sim = (sim_matrix * (1 - mask)).sum(dim=1) / (1 - mask).sum(dim=1).clamp(min=1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
        return loss
    
    def cluster_consistency_loss(self, posteriors):
        """Encourage compact and separated clusters."""
        entropy = -(posteriors * torch.log(posteriors + 1e-8)).sum(dim=1).mean()
        avg_posterior = posteriors.mean(dim=0)
        diversity = -(avg_posterior * torch.log(avg_posterior + 1e-8)).sum()
        return entropy - 0.1 * diversity
    
    def monotonic_loss(self, model):
        """Regularization for monotonic constraints."""
        loss = 0
        regressor = model.regressor
        
        # Check if MLP with monotonic constraint
        if hasattr(regressor, 'fc3') and hasattr(regressor, 'positive_transform'):
            weight = regressor.fc3.weight
            loss = F.relu(-weight).sum()
        
        return loss
    
    def forward(self, outputs, targets, model):
        """
        Compute total loss.
        Args:
            outputs: Dict with model outputs
            targets: Dict with ground truth
            model: CENIQA model
        Returns:
            dict with loss components
        """
        losses = {}
        
        # Quality loss
        if 'quality_scores' in targets:
            losses['quality'] = self.mse_loss(outputs['quality_score'], targets['quality_scores'])
        
        # Contrastive loss
        if self.config.lambda_contrast > 0 and 'distortion_labels' in targets:
            losses['contrast'] = self.contrastive_loss(outputs['features'], targets['distortion_labels'])
        
        # Distortion consistency loss
        if self.config.lambda_consistency > 0 and 'distortion_labels' in targets:
            losses['consistency'] = self.ce_loss(outputs['distortion_logits'], targets['distortion_labels'])
        
        # Ranking loss
        if self.config.lambda_ranking > 0 and 'ranking_pairs' in targets:
            idx1, idx2 = targets['ranking_pairs']
            score1 = outputs['ranking_score'][idx1]
            score2 = outputs['ranking_score'][idx2]
            ranking_target = torch.ones(len(idx1), device=score1.device)
            losses['ranking'] = self.ranking_loss(score1, score2, ranking_target)
        
        # Cluster consistency loss
        if self.config.lambda_cluster > 0:
            losses['cluster'] = self.cluster_consistency_loss(outputs['posteriors'])
        
        # Monotonic regularization
        if self.config.lambda_monotonic > 0 and self.config.use_monotonic:
            losses['monotonic'] = self.monotonic_loss(model)
        
        # Combine losses
        total_loss = 0
        for key, loss in losses.items():
            weight = getattr(self.config, f'lambda_{key}', 1.0)
            total_loss += weight * loss
        
        losses['total'] = total_loss
        return losses
