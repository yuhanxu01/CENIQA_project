"""Gaussian Mixture Model for clustering distortions."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


class DifferentiableGMM(nn.Module):
    """Learnable Gaussian Mixture Model for end-to-end training."""
    def __init__(self, n_clusters, feature_dim, covariance_type='diag'):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.covariance_type = covariance_type
        
        # Learnable parameters
        self.means = nn.Parameter(torch.randn(n_clusters, feature_dim))
        self.log_vars = nn.Parameter(torch.zeros(n_clusters, feature_dim))
        self.log_weights = nn.Parameter(torch.zeros(n_clusters))
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.means, -1, 1)
        
    def forward(self, x):
        """
        Compute GMM posterior probabilities.
        Args:
            x: [batch_size, feature_dim]
        Returns:
            posteriors: [batch_size, n_clusters]
        """
        batch_size = x.shape[0]
        log_probs = []
        
        for k in range(self.n_clusters):
            # Mahalanobis distance
            diff = x - self.means[k].unsqueeze(0)
            var = torch.exp(self.log_vars[k])
            
            # Log probability under Gaussian
            log_prob = -0.5 * torch.sum(diff**2 / var, dim=1)
            log_prob -= 0.5 * torch.sum(self.log_vars[k])
            log_prob -= 0.5 * self.feature_dim * np.log(2 * np.pi)
            
            # Add log mixing weight
            log_prob += F.log_softmax(self.log_weights, dim=0)[k]
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)
        posteriors = F.softmax(log_probs, dim=1)
        
        return posteriors
    
    def fit_sklearn(self, features):
        """
        Initialize with sklearn GMM for better starting point.
        Args:
            features: numpy array [N, feature_dim]
        """
        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            init_params='k-means++'
        )
        gmm.fit(features)
        
        # Copy parameters
        self.means.data = torch.from_numpy(gmm.means_).float()
        if self.covariance_type == 'diag':
            self.log_vars.data = torch.log(torch.from_numpy(gmm.covariances_).float())
        self.log_weights.data = torch.log(torch.from_numpy(gmm.weights_).float())
    
    def get_cluster_assignments(self, x):
        """Get hard cluster assignments."""
        posteriors = self.forward(x)
        assignments = torch.argmax(posteriors, dim=1)
        return assignments
