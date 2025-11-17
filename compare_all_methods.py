"""HPC一键比较7个方法：No GMM + 标准GMM + 5种GMM改进方法"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbones import build_backbone
from gmm_module import DifferentiableGMM
from regressors import build_regressor
from config import ModelConfig
from high_res_distorted_dataset_lazy import LazyHighResDistortedDataset
import os
from datetime import datetime
import json
import argparse
import time


# ============== Baseline: No GMM ==============
class NoGMMCENIQA(nn.Module):
    """Baseline: 没有GMM，只有backbone + regressor"""
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

    def forward(self, x, return_all=False):
        features = self.backbone(x)
        features = self.feature_proj(features)
        quality_score = self.regressor(features).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'features': features
            }
        return quality_score


# ============== 标准GMM (当前实现) ==============
class StandardGMMCENIQA(nn.Module):
    """标准GMM: 当前的baseline实现 - sklearn GMM + concatenate posteriors"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # 标准GMM
        self.gmm = DifferentiableGMM(
            config.n_clusters,
            config.hidden_dim,
            config.gmm_covariance_type
        )

        # 简单concatenate后回归
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

        # 简单concatenate
        combined = torch.cat([features, posteriors], dim=-1)
        quality_score = self.regressor(combined).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'features': features,
                'posteriors': posteriors
            }
        return quality_score


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
def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        scores = batch['score'].to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(images, return_all=True) if hasattr(model, 'experts') or hasattr(model, 'gmm') else model(images)

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

        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch+1}/{total_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} Time: {elapsed:.1f}s")

    return total_loss / total_samples


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            scores = batch['score'].to(device)

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
    print(f"\n{'='*80}")
    print(f"训练模型: {model_name}")
    print(f"{'='*80}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_srcc = -1
    results = {
        'model_name': model_name,
        'train_losses': [],
        'val_srcc': [],
        'val_plcc': [],
        'best_srcc': -1,
        'best_plcc': -1,
        'best_epoch': -1
    }

    for epoch in range(epochs):
        print(f"\n{'-'*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'-'*80}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
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
            results['best_epoch'] = epoch + 1
            # 保存最佳模型
            save_path = f'checkpoints/{model_name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'srcc': val_srcc,
                'plcc': val_plcc,
            }, save_path)
            print(f"✓ Saved best model: SRCC={val_srcc:.4f}, PLCC={val_plcc:.4f}")

        scheduler.step(val_srcc)

    print(f"\n{'='*80}")
    print(f"{model_name} 训练完成!")
    print(f"最佳结果 (Epoch {results['best_epoch']}): SRCC={results['best_srcc']:.4f}, PLCC={results['best_plcc']:.4f}")
    print(f"{'='*80}\n")

    return results


def main():
    """主函数 - 一键运行比较"""
    parser = argparse.ArgumentParser(description='比较7种IQA方法')
    parser.add_argument('--test_mode', action='store_true', help='快速测试模式：少量数据+2epochs')
    parser.add_argument('--epochs', type=int, default=50, help='训练epochs数')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_train', type=int, default=5000, help='训练样本数')
    parser.add_argument('--num_val', type=int, default=1000, help='验证样本数')
    parser.add_argument('--data_root', type=str, default='data/stl10', help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='.', help='输出目录')
    args = parser.parse_args()

    # 测试模式覆盖参数
    if args.test_mode:
        print("\n" + "!"*80)
        print("! 测试模式: 使用少量数据和2个epochs验证代码和环境")
        print("!"*80 + "\n")
        args.epochs = 2
        args.num_train = 100
        args.num_val = 50
        args.batch_size = 8

    print("="*80)
    print("HPC一键比较7种方法：No GMM + 标准GMM + 5种GMM改进")
    print("="*80)
    print(f"\n配置:")
    print(f"  - 训练样本: {args.num_train}")
    print(f"  - 验证样本: {args.num_val}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 数据路径: {args.data_root}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 配置
    config = ModelConfig()
    config.n_clusters = 5
    config.hidden_dim = 512
    config.dropout_rate = 0.2
    config.backbone = 'resnet50'
    config.pretrained = True
    config.regressor_type = 'mlp'

    # 创建目录
    os.makedirs('checkpoints', exist_ok=True)

    # 数据集
    print("\n加载数据集...")
    train_dataset = LazyHighResDistortedDataset(
        root_dir=args.data_root,
        split='train',
        num_samples=args.num_train,
        distortion_types=['blur', 'noise', 'jpeg', 'contrast'],
        cache_size=min(200, args.num_train)
    )

    val_dataset = LazyHighResDistortedDataset(
        root_dir=args.data_root,
        split='test',
        num_samples=args.num_val,
        distortion_types=['blur', 'noise', 'jpeg', 'contrast'],
        cache_size=min(100, args.num_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本")

    # 定义7个模型
    models = {
        '0_NoGMM': NoGMMCENIQA(config),
        '1_StandardGMM': StandardGMMCENIQA(config),
        '2_MoE': MoECENIQA(config),
        '3_Attention': AttentionGatedCENIQA(config),
        '4_LearnableGMM': LearnableGMMCENIQA(config),
        '5_DistortionAware': DistortionAwareCENIQA(config),
        '6_Complete': CompleteCENIQA(config)
    }

    print(f"\n共 {len(models)} 个模型待训练:")
    for i, name in enumerate(models.keys(), 1):
        print(f"  {i}. {name}")

    # 训练所有模型
    all_results = {}
    total_start = time.time()

    for model_name, model in models.items():
        model_start = time.time()

        results = train_and_evaluate(
            model_name, model, train_loader, val_loader,
            device, epochs=args.epochs, lr=args.lr
        )
        all_results[model_name] = results

        model_time = time.time() - model_start
        print(f"\n{model_name} 总耗时: {model_time/60:.2f} 分钟\n")

    total_time = time.time() - total_start

    # 打印最终比较
    print("\n" + "="*80)
    print("最终结果比较")
    print("="*80)
    print(f"{'模型':<25} {'最佳SRCC':<12} {'最佳PLCC':<12} {'最佳Epoch':<12}")
    print("-"*80)

    # 按SRCC排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_srcc'], reverse=True)

    for rank, (model_name, results) in enumerate(sorted_results, 1):
        marker = "🏆" if rank == 1 else f"{rank}."
        print(f"{marker} {model_name:<22} {results['best_srcc']:<12.4f} "
              f"{results['best_plcc']:<12.4f} {results['best_epoch']:<12}")

    print("-"*80)
    print(f"总耗时: {total_time/60:.2f} 分钟 ({total_time/3600:.2f} 小时)")
    print("="*80)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_test" if args.test_mode else "_full"
    results_file = os.path.join(args.output_dir, f'comparison_results{mode_suffix}_{timestamp}.json')

    # 添加配置信息
    save_data = {
        'config': {
            'test_mode': args.test_mode,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'num_train': args.num_train,
            'num_val': args.num_val,
            'total_time_minutes': total_time/60,
        },
        'results': all_results
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✓ 结果已保存到: {results_file}")

    if args.test_mode:
        print("\n" + "!"*80)
        print("! 测试模式完成! 如果一切正常，请用完整数据重新运行:")
        print("! python compare_all_methods.py --epochs 50 --num_train 5000 --num_val 1000")
        print("!"*80)
    else:
        print("\n🎉 完整实验完成!")


if __name__ == '__main__':
    main()
