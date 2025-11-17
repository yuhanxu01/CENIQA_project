"""
单个方法训练脚本 - 用于HPC并行提交
支持7个方案的独立训练：5个GMM改进 + No GMM baseline + 单纯GMM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from backbones import build_backbone
from gmm_module import DifferentiableGMM
from regressors import build_regressor
from config import ModelConfig
from high_res_distorted_dataset_lazy import HighResDistortedDatasetLazy
import os
import argparse
from datetime import datetime
import json
from scipy.stats import spearmanr, pearsonr
import sys


# ============== Baseline: No GMM ==============
class NoGMMCENIQA(nn.Module):
    """Baseline: 不使用GMM，直接回归"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # 直接回归，不使用GMM
        self.regressor = build_regressor(
            config.regressor_type,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout_rate
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_proj(features)
        quality_score = self.regressor(features).squeeze(-1)
        return quality_score


# ============== 单纯GMM版本 ==============
class VanillaGMMCENIQA(nn.Module):
    """单纯GMM版本：GMM + 简单拼接"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # 标准GMM
        self.gmm = DifferentiableGMM(config.n_clusters, config.hidden_dim, config.gmm_covariance_type)

        # 简单拼接features和posteriors后回归
        regressor_input_dim = config.hidden_dim + config.n_clusters
        self.regressor = build_regressor(
            config.regressor_type,
            regressor_input_dim,
            config.hidden_dim,
            config.dropout_rate
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_proj(features)
        posteriors = self.gmm(features)

        # 简单拼接
        combined = torch.cat([features, posteriors], dim=-1)
        quality_score = self.regressor(combined).squeeze(-1)
        return quality_score


# ============== 方案1: Mixture of Expert Regressors ==============
class MoECENIQA(nn.Module):
    """方案1: MoE - 每个cluster一个expert regressor"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.gmm = DifferentiableGMM(config.n_clusters, config.hidden_dim, config.gmm_covariance_type)

        self.experts = nn.ModuleList([
            build_regressor(config.regressor_type, config.hidden_dim, config.hidden_dim, config.dropout_rate)
            for _ in range(config.n_clusters)
        ])

        self.gating = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_clusters),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_proj(features)
        posteriors = self.gmm(features)
        gates = self.gating(features)
        weights = posteriors * gates
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        expert_predictions = []
        for expert in self.experts:
            pred = expert(features)
            expert_predictions.append(pred)

        expert_predictions = torch.stack(expert_predictions, dim=1)
        quality_score = torch.sum(weights.unsqueeze(-1) * expert_predictions, dim=1).squeeze(-1)
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

        self.cluster_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
            for _ in range(config.n_clusters)
        ])

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

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_proj(features)
        posteriors = self.gmm(features)

        cluster_features = []
        for proj in self.cluster_projections:
            cf = proj(features)
            cluster_features.append(cf)

        cluster_features = torch.stack(cluster_features, dim=1)
        features_expanded = features.unsqueeze(1)
        attended_features, _ = self.cross_attn(
            features_expanded,
            cluster_features,
            cluster_features
        )

        weighted_features = cluster_features * posteriors.unsqueeze(-1)
        final_features = weighted_features.sum(dim=1) + attended_features.squeeze(1)
        quality_score = self.regressor(final_features).squeeze(-1)
        return quality_score


# ============== 方案3: Learnable GMM ==============
class LearnableGMM(nn.Module):
    """Learnable GMM - 参数由网络预测"""
    def __init__(self, feature_dim, n_clusters, hidden_dim=256):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim

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
        self.gmm = LearnableGMM(config.hidden_dim, config.n_clusters, config.hidden_dim)

        regressor_input_dim = config.hidden_dim + config.n_clusters
        self.regressor = build_regressor(
            config.regressor_type,
            regressor_input_dim,
            config.hidden_dim,
            config.dropout_rate
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_proj(features)
        posteriors = self.gmm(features)
        combined = torch.cat([features, posteriors], dim=-1)
        quality_score = self.regressor(combined).squeeze(-1)
        return quality_score


# ============== 方案4: Distortion-Aware Multi-Expert ==============
class DistortionAwareCENIQA(nn.Module):
    """方案4: Distortion-Aware - 显式建模distortion types"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)

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
        self.distortion_classifier = nn.Linear(256, config.n_clusters)
        self.gmm = DifferentiableGMM(config.n_clusters, config.hidden_dim, config.gmm_covariance_type)

        self.quality_experts = nn.ModuleList([
            build_regressor(config.regressor_type, config.hidden_dim, config.hidden_dim, config.dropout_rate)
            for _ in range(config.n_clusters)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + 256 + config.n_clusters, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, x):
        content_features = self.backbone(x)
        content_features = self.feature_proj(content_features)

        distortion_features = self.distortion_encoder(x)
        distortion_logits = self.distortion_classifier(distortion_features)
        distortion_probs = F.softmax(distortion_logits, dim=-1)

        posteriors = self.gmm(content_features)

        expert_outputs = []
        for expert in self.quality_experts:
            pred = expert(content_features)
            expert_outputs.append(pred)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        quality_from_experts = torch.sum(
            distortion_probs.unsqueeze(-1) * expert_outputs,
            dim=1
        )

        combined = torch.cat([content_features, distortion_features, posteriors], dim=-1)
        quality_from_fusion = self.fusion(combined)

        quality_score = (quality_from_experts + quality_from_fusion).squeeze(-1) / 2
        return quality_score


# ============== 方案5: 完整的Self-Supervised Pipeline ==============
class CompleteCENIQA(nn.Module):
    """方案5: 完整的Self-Supervised GMM-IQA Pipeline"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.gmm = LearnableGMM(config.hidden_dim, config.n_clusters, config.hidden_dim)

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

        self.experts = nn.ModuleList([
            build_regressor(config.regressor_type, config.hidden_dim, config.hidden_dim, config.dropout_rate)
            for _ in range(config.n_clusters)
        ])

        self.gating = nn.Sequential(
            nn.Linear(config.hidden_dim + 256, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_clusters),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + 256 + config.n_clusters, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, x):
        content_features = self.backbone(x)
        content_features = self.feature_proj(content_features)
        distortion_features = self.distortion_encoder(x)
        posteriors = self.gmm(content_features)

        combined_for_gating = torch.cat([content_features, distortion_features], dim=-1)
        gates = self.gating(combined_for_gating)
        weights = posteriors * gates
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        expert_outputs = []
        for expert in self.experts:
            pred = expert(content_features)
            expert_outputs.append(pred)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        quality_from_experts = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)

        combined_for_fusion = torch.cat([content_features, distortion_features, posteriors], dim=-1)
        quality_from_fusion = self.fusion(combined_for_fusion)

        quality_score = (quality_from_experts + quality_from_fusion).squeeze(-1) / 2
        return quality_score


# ============== 训练和评估函数 ==============
def train_epoch(model, dataloader, optimizer, device, print_freq=50):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            scores = batch['score'].to(device)
        else:
            images, scores = batch
            images = images.to(device)
            scores = scores.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, dict):
            pred_scores = outputs['quality_score']
        else:
            pred_scores = outputs

        loss = F.mse_loss(pred_scores, scores)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % print_freq == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / total_samples


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
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

    srcc = spearmanr(all_preds.numpy(), all_targets.numpy())[0]
    plcc = pearsonr(all_preds.numpy(), all_targets.numpy())[0]

    return srcc, plcc


def get_model(method_name, config):
    """根据方法名获取模型"""
    models = {
        'no_gmm': NoGMMCENIQA,
        'vanilla_gmm': VanillaGMMCENIQA,
        'moe': MoECENIQA,
        'attention': AttentionGatedCENIQA,
        'learnable_gmm': LearnableGMMCENIQA,
        'distortion_aware': DistortionAwareCENIQA,
        'complete': CompleteCENIQA
    }

    if method_name not in models:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(models.keys())}")

    return models[method_name](config)


def main():
    parser = argparse.ArgumentParser(description='训练单个IQA方法')
    parser.add_argument('--method', type=str, required=True,
                      choices=['no_gmm', 'vanilla_gmm', 'moe', 'attention',
                               'learnable_gmm', 'distortion_aware', 'complete'],
                      help='训练方法')
    parser.add_argument('--epochs', type=int, default=60,
                      help='训练epochs数')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--max_train_samples', type=int, default=None,
                      help='最大训练样本数（用于快速测试）')
    parser.add_argument('--max_val_samples', type=int, default=None,
                      help='最大验证样本数（用于快速测试）')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='模型保存目录')
    parser.add_argument('--quick_test', action='store_true',
                      help='快速测试模式')

    args = parser.parse_args()

    # 快速测试模式
    if args.quick_test:
        args.epochs = 2
        args.max_train_samples = 500
        args.max_val_samples = 200
        print("\n" + "="*80)
        print("快速测试模式 - 2 epochs, 500训练样本, 200验证样本")
        print("="*80 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"训练方法: {args.method}")

    # 配置
    config = ModelConfig()
    config.n_clusters = 5
    config.hidden_dim = 512
    config.dropout_rate = 0.2
    config.backbone = 'resnet50'
    config.pretrained = True

    # 数据集
    print("\n加载数据集...")
    full_train_dataset = HighResDistortedDatasetLazy(
        dataset_name='stl10',
        split='train',
        max_samples=args.max_train_samples,
        distortions_per_image=5,
        include_pristine=True,
        distortion_strength='medium'
    )

    full_test_dataset = HighResDistortedDatasetLazy(
        dataset_name='stl10',
        split='test',
        max_samples=args.max_val_samples,
        distortions_per_image=5,
        include_pristine=True,
        distortion_strength='medium'
    )

    # 合并并90/10分割
    combined_dataset = ConcatDataset([full_train_dataset, full_test_dataset])
    total_size = len(combined_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    print(f"\n数据集总计: {total_size} 样本")
    print(f"训练集: {train_size} 样本 (90%)")
    print(f"验证集: {val_size} 样本 (10%)")

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

    # 创建模型
    print(f"\n创建模型: {args.method}")
    model = get_model(args.method, config)
    model = model.to(device)

    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # 训练
    print(f"\n{'='*80}")
    print(f"开始训练 - {args.method}")
    print(f"{'='*80}\n")

    best_srcc = -1
    results = {
        'method': args.method,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'train_size': train_size,
        'val_size': val_size,
        'train_losses': [],
        'val_srcc': [],
        'val_plcc': [],
        'best_srcc': -1,
        'best_plcc': -1,
        'best_epoch': -1
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # 评估
        val_srcc, val_plcc = evaluate(model, val_loader, device)
        print(f"Validation SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}")

        # 记录
        results['train_losses'].append(float(train_loss))
        results['val_srcc'].append(float(val_srcc))
        results['val_plcc'].append(float(val_plcc))

        # 保存最佳模型
        if val_srcc > best_srcc:
            best_srcc = val_srcc
            results['best_srcc'] = float(val_srcc)
            results['best_plcc'] = float(val_plcc)
            results['best_epoch'] = epoch + 1

            checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.method}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_srcc': best_srcc,
                'best_plcc': val_plcc,
            }, checkpoint_path)
            print(f"✓ 保存最佳模型到: {checkpoint_path}")

        scheduler.step(val_srcc)

    # 保存结果
    print(f"\n{'='*80}")
    print(f"训练完成 - {args.method}")
    print(f"最佳结果: SRCC={results['best_srcc']:.4f}, PLCC={results['best_plcc']:.4f}, Epoch={results['best_epoch']}")
    print(f"{'='*80}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'{args.method}_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"结果已保存到: {results_file}")


if __name__ == '__main__':
    main()
