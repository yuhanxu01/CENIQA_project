"""Regression heads for quality score prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicMLP(nn.Module):
    """MLP with monotonic constraints for quality prediction."""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.positive_transform = nn.Softplus()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Positive constraint for monotonicity
        weight = self.positive_transform(self.fc3.weight)
        x = F.linear(x, weight, self.fc3.bias)
        
        return torch.sigmoid(x)


class KANLayer(nn.Module):
    """Single KAN layer with learnable activation functions."""
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        self.spline_coeffs = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        self.spline_weight = nn.Parameter(torch.ones(out_features, in_features))
        
    def forward(self, x):
        # Normalize input
        x_normalized = torch.sigmoid(x)
        x_expanded = x_normalized.unsqueeze(1).unsqueeze(-1)
        
        # B-spline basis
        grid_points = torch.linspace(0, 1, self.grid_size, device=x.device)
        distances = torch.abs(x_expanded - grid_points)
        weights = F.softmax(-distances * 10, dim=-1)
        
        # Spline transformation
        spline_out = torch.einsum('boig,oig->boi', weights, self.spline_coeffs)
        weighted_out = spline_out * self.spline_weight.unsqueeze(0)
        
        output = weighted_out.sum(dim=2)
        return output


class KANRegressor(nn.Module):
    """Kolmogorov-Arnold Network for regression."""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.layer1 = KANLayer(input_dim, hidden_dim)
        self.layer2 = KANLayer(hidden_dim, hidden_dim // 2)
        self.layer3 = KANLayer(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return torch.sigmoid(x)


class TransformerRegressor(nn.Module):
    """Transformer-based regression head."""
    def __init__(self, input_dim, hidden_dim=256, n_heads=8, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = self.output_proj(x).squeeze(1)
        return torch.sigmoid(x)


class GRURegressor(nn.Module):
    """GRU-based regression head."""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.gru(x)
        x = self.output_proj(h.squeeze(0))
        return torch.sigmoid(x)


class AttentionRegressor(nn.Module):
    """Attention-based regression with cluster weighting."""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        x_proj = self.feature_proj(x).unsqueeze(1)
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        out = self.mlp(attn_out.squeeze(1))
        return torch.sigmoid(out)


def build_regressor(regressor_type: str, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
    """Factory function to build regressor."""
    if regressor_type == 'mlp':
        return MonotonicMLP(input_dim, hidden_dim, dropout)
    elif regressor_type == 'kan':
        return KANRegressor(input_dim, hidden_dim, dropout)
    elif regressor_type == 'transformer':
        return TransformerRegressor(input_dim, hidden_dim, dropout=dropout)
    elif regressor_type == 'gru':
        return GRURegressor(input_dim, hidden_dim, dropout)
    elif regressor_type == 'attention':
        return AttentionRegressor(input_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")
