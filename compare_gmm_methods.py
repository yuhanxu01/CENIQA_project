"""一键比较5种GMM改进方法"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbones import build_backbone
from gmm_module import DifferentiableGMM
from regressors import build_regressor
from config import ModelConfig
from high_res_distorted_dataset_lazy import HighResDistortedDatasetLazy
import os
from datetime import datetime
import json


# ============== 方案1: Mixture of Expert Regressors ==============
class MoECENIQA(nn.Module):
    """方案1: MoE - 每个cluster一个expert regressor"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # GMM for clustering
        self.gmm = DifferentiableGMM(config.n_clusters, config.hidden_dim, config.gmm_covariance_type)

        # 每个cluster一个expert regressor
        self.experts = nn.ModuleList([
            build_regressor(config.regressor_type, config.hidden_dim, config.hidden_dim, config.dropout_rate)
            for _ in range(config.n_clusters)
        ])

        # Gating network
        self.gating = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_clusters),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, return_all=False):
        features = self.backbone(x)
        features = self.feature_proj(features)

        # Get cluster posteriors
        posteriors = self.gmm(features)

        # Gating weights
        gates = self.gating(features)
        weights = posteriors * gates
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # 每个expert预测
        expert_predictions = []
        for expert in self.experts:
            pred = expert(features)
            expert_predictions.append(pred)

        expert_predictions = torch.stack(expert_predictions, dim=1)  # [B, K, 1]

        # Weighted combination
        quality_score = torch.sum(weights.unsqueeze(-1) * expert_predictions, dim=1).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'posteriors': posteriors,
                'gates': gates,
                'expert_predictions': expert_predictions
            }
        return quality_score


# ============== 方案2: Attention-Gated Feature Fusion ==============
class AttentionGatedCENIQA(nn.Module):
    """方案2: Attention机制 - posteriors调制features"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.gmm = DifferentiableGMM(config.n_clusters, config.hidden_dim, config.gmm_covariance_type)

        # Cluster-specific feature transformations
        self.cluster_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
            for _ in range(config.n_clusters)
        ])

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )

        self.regressor = build_regressor(
            config.regressor_type,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout_rate
        )

    def forward(self, x, return_all=False):
        features = self.backbone(x)
        features = self.feature_proj(features)
        posteriors = self.gmm(features)

        # 每个cluster transform features
        cluster_features = []
        for proj in self.cluster_projections:
            cf = proj(features)
            cluster_features.append(cf)

        cluster_features = torch.stack(cluster_features, dim=1)  # [B, K, D]

        # Attention
        features_expanded = features.unsqueeze(1)
        attended_features, _ = self.cross_attn(
            features_expanded,
            cluster_features,
            cluster_features
        )

        # Combine with posterior weights
        weighted_features = cluster_features * posteriors.unsqueeze(-1)
        final_features = weighted_features.sum(dim=1) + attended_features.squeeze(1)

        quality_score = self.regressor(final_features).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'posteriors': posteriors,
                'features': final_features
            }
        return quality_score


# ============== 方案3: Learnable GMM ==============
class LearnableGMM(nn.Module):
    """Learnable GMM - 参数由网络预测"""
    def __init__(self, feature_dim, n_clusters, hidden_dim=256):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim

        # Network to predict GMM parameters
        self.param_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, n_clusters * feature_dim)
        self.logvar_head = nn.Linear(hidden_dim, n_clusters * feature_dim)
        self.weight_head = nn.Linear(hidden_dim, n_clusters)

    def forward(self, x):
        B, D = x.shape
        h = self.param_network(x)

        means = self.mean_head(h).view(B, self.n_clusters, D)
        log_vars = self.logvar_head(h).view(B, self.n_clusters, D)
        log_weights = self.weight_head(h)

        # Compute posteriors
        log_probs = []
        for k in range(self.n_clusters):
            diff = x.unsqueeze(1) - means[:, k:k+1, :]
            var = torch.exp(log_vars[:, k, :]) + 1e-6

            log_prob = -0.5 * torch.sum(diff**2 / var.unsqueeze(1), dim=-1)
            log_prob -= 0.5 * torch.sum(log_vars[:, k, :], dim=-1, keepdim=True)
            log_prob += F.log_softmax(log_weights, dim=-1)[:, k:k+1]

            log_probs.append(log_prob)

        log_probs = torch.cat(log_probs, dim=1)
        posteriors = F.softmax(log_probs, dim=1)

        return posteriors


class LearnableGMMCENIQA(nn.Module):
    """方案3: Learnable GMM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # Learnable GMM
        self.gmm = LearnableGMM(config.hidden_dim, config.n_clusters, config.hidden_dim)

        # Regressor
        regressor_input_dim = config.hidden_dim + config.n_clusters
        self.regressor = build_regressor(
            config.regressor_type,
            regressor_input_dim,
            config.hidden_dim,
            config.dropout_rate
        )

    def forward(self, x, return_all=False):
        features = self.backbone(x)
        features = self.feature_proj(features)
        posteriors = self.gmm(features)

        combined = torch.cat([features, posteriors], dim=-1)
        quality_score = self.regressor(combined).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'posteriors': posteriors,
                'features': features
            }
        return quality_score


# ============== 方案4: Distortion-Aware Multi-Expert ==============
class DistortionAwareCENIQA(nn.Module):
    """方案4: Distortion-Aware - 显式建模distortion types"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)

        # Distortion-aware feature extractor
        self.distortion_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256)
        )

        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # Distortion classifier
        self.distortion_classifier = nn.Linear(256, config.n_clusters)

        # GMM
        self.gmm = DifferentiableGMM(config.n_clusters, config.hidden_dim, config.gmm_covariance_type)

        # Distortion-specific quality experts
        self.quality_experts = nn.ModuleList([
            build_regressor(config.regressor_type, config.hidden_dim, config.hidden_dim, config.dropout_rate)
            for _ in range(config.n_clusters)
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + 256 + config.n_clusters, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, x, return_all=False):
        content_features = self.backbone(x)
        content_features = self.feature_proj(content_features)

        # Distortion features
        distortion_features = self.distortion_encoder(x)
        distortion_logits = self.distortion_classifier(distortion_features)
        distortion_probs = F.softmax(distortion_logits, dim=-1)

        # GMM posteriors
        posteriors = self.gmm(content_features)

        # Distortion-specific quality prediction
        expert_outputs = []
        for expert in self.quality_experts:
            pred = expert(content_features)
            expert_outputs.append(pred)

        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Weighted by distortion classification
        quality_from_experts = torch.sum(
            distortion_probs.unsqueeze(-1) * expert_outputs,
            dim=1
        )

        # Fusion
        combined = torch.cat([content_features, distortion_features, posteriors], dim=-1)
        quality_from_fusion = self.fusion(combined)

        # Final prediction
        quality_score = (quality_from_experts + quality_from_fusion).squeeze(-1) / 2

        if return_all:
            return {
                'quality_score': quality_score,
                'distortion_logits': distortion_logits,
                'posteriors': posteriors,
                'expert_outputs': expert_outputs
            }
        return quality_score


# ============== 方案5: 完整的Self-Supervised Pipeline ==============
class CompleteCENIQA(nn.Module):
    """方案5: 完整的Self-Supervised GMM-IQA Pipeline"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # Learnable GMM with contrastive learning
        self.gmm = LearnableGMM(config.hidden_dim, config.n_clusters, config.hidden_dim)

        # Distortion encoder
        self.distortion_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256)
        )

        # MoE regressors
        self.experts = nn.ModuleList([
            build_regressor(config.regressor_type, config.hidden_dim, config.hidden_dim, config.dropout_rate)
            for _ in range(config.n_clusters)
        ])

        # Gating with attention
        self.gating = nn.Sequential(
            nn.Linear(config.hidden_dim + 256, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_clusters),
            nn.Softmax(dim=-1)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + 256 + config.n_clusters, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, x, return_all=False):
        content_features = self.backbone(x)
        content_features = self.feature_proj(content_features)

        # Distortion features
        distortion_features = self.distortion_encoder(x)

        # GMM posteriors
        posteriors = self.gmm(content_features)

        # Gating
        combined_for_gating = torch.cat([content_features, distortion_features], dim=-1)
        gates = self.gating(combined_for_gating)
        weights = posteriors * gates
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Expert predictions
        expert_outputs = []
        for expert in self.experts:
            pred = expert(content_features)
            expert_outputs.append(pred)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        quality_from_experts = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)

        # Fusion
        combined_for_fusion = torch.cat([content_features, distortion_features, posteriors], dim=-1)
        quality_from_fusion = self.fusion(combined_for_fusion)

        # Ensemble
        quality_score = (quality_from_experts + quality_from_fusion).squeeze(-1) / 2

        if return_all:
            return {
                'quality_score': quality_score,
                'posteriors': posteriors,
                'gates': gates,
                'expert_outputs': expert_outputs
            }
        return quality_score


# ============== 训练和评估 ==============
def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        # Handle both tuple and dict formats
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            scores = batch['score'].to(device)
        else:
            images, scores = batch
            images = images.to(device)
            scores = scores.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(images, return_all=True) if hasattr(model, 'experts') else model(images)

        if isinstance(outputs, dict):
            pred_scores = outputs['quality_score']
        else:
            pred_scores = outputs

        # Loss
        loss = F.mse_loss(pred_scores, scores)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / total_samples


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both tuple and dict formats
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                scores = batch['score'].to(device)
            else:
                images, scores = batch
                images = images.to(device)
                scores = scores.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                preds = outputs['quality_score']
            else:
                preds = outputs

            all_preds.append(preds.cpu())
            all_targets.append(scores.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # 计算SRCC和PLCC
    from scipy.stats import spearmanr, pearsonr
    srcc = spearmanr(all_preds.numpy(), all_targets.numpy())[0]
    plcc = pearsonr(all_preds.numpy(), all_targets.numpy())[0]

    return srcc, plcc


def train_and_evaluate(model_name, model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    """训练并评估一个模型"""
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_srcc = -1
    results = {
        'model_name': model_name,
        'train_losses': [],
        'val_srcc': [],
        'val_plcc': [],
        'best_srcc': -1,
        'best_plcc': -1
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # 评估
        val_srcc, val_plcc = evaluate(model, val_loader, device)
        print(f"Validation SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}")

        # 记录
        results['train_losses'].append(train_loss)
        results['val_srcc'].append(val_srcc)
        results['val_plcc'].append(val_plcc)

        # 更新最佳
        if val_srcc > best_srcc:
            best_srcc = val_srcc
            results['best_srcc'] = val_srcc
            results['best_plcc'] = val_plcc
            # 保存最佳模型
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')

        scheduler.step(val_srcc)

    print(f"\n{model_name} 最佳结果: SRCC={results['best_srcc']:.4f}, PLCC={results['best_plcc']:.4f}")
    return results


def main():
    """主函数 - 一键运行比较"""
    print("="*80)
    print("一键比较5种GMM改进方法")
    print("="*80)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 配置
    config = ModelConfig()
    config.n_clusters = 5  # 5个clusters
    config.hidden_dim = 512
    config.dropout_rate = 0.2
    config.backbone = 'resnet50'
    config.pretrained = True

    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)

    # 数据集
    print("\n加载数据集...")
    train_dataset = HighResDistortedDatasetLazy(
        dataset_name='stl10',
        split='train',
        max_samples=500,  # 减少样本数以加快测试
        distortions_per_image=4,
        include_pristine=True,
        distortion_strength='medium'
    )

    val_dataset = HighResDistortedDatasetLazy(
        dataset_name='stl10',
        split='test',
        max_samples=200,
        distortions_per_image=4,
        include_pristine=True,
        distortion_strength='medium'
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 定义5个模型
    models = {
        '方案1_MoE': MoECENIQA(config),
        '方案2_Attention': AttentionGatedCENIQA(config),
        '方案3_LearnableGMM': LearnableGMMCENIQA(config),
        '方案4_DistortionAware': DistortionAwareCENIQA(config),
        '方案5_Complete': CompleteCENIQA(config)
    }

    # 训练参数
    epochs = 15
    lr = 1e-4

    # 训练所有模型并记录结果
    all_results = {}

    for model_name, model in models.items():
        results = train_and_evaluate(
            model_name, model, train_loader, val_loader,
            device, epochs=epochs, lr=lr
        )
        all_results[model_name] = results

    # 打印最终比较
    print("\n" + "="*80)
    print("最终结果比较")
    print("="*80)
    print(f"{'模型':<25} {'最佳SRCC':<12} {'最佳PLCC':<12}")
    print("-"*80)

    for model_name, results in all_results.items():
        print(f"{model_name:<25} {results['best_srcc']:<12.4f} {results['best_plcc']:<12.4f}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'gmm_comparison_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存到: {results_file}")
    print("\n完成！🎉")


if __name__ == '__main__':
    main()
