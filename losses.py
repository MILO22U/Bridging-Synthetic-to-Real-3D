"""
Loss functions for 3D point cloud reconstruction.

Chamfer Distance: The standard loss for point cloud comparison.
Measures the average nearest-neighbor distance between two point sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    bidirectional: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Chamfer Distance between two point clouds.
    
    CD(P, Q) = (1/|P|) Σ_{p∈P} min_{q∈Q} ||p - q||² 
             + (1/|Q|) Σ_{q∈Q} min_{p∈P} ||q - p||²
    
    Args:
        pred:   (B, N, 3) predicted point cloud
        target: (B, M, 3) ground truth point cloud
        bidirectional: If True, compute both directions
    
    Returns:
        cd_loss: scalar, total Chamfer Distance
        cd_p2t:  scalar, pred → target direction
        cd_t2p:  scalar, target → pred direction
    """
    # Compute pairwise distances: (B, N, M)
    # ||p - q||² = ||p||² + ||q||² - 2 * p·q
    pred_sq = (pred ** 2).sum(dim=2, keepdim=True)       # (B, N, 1)
    target_sq = (target ** 2).sum(dim=2, keepdim=True)   # (B, M, 1)
    
    # (B, N, M)
    dist_matrix = pred_sq + target_sq.transpose(1, 2) - 2 * torch.bmm(pred, target.transpose(1, 2))
    dist_matrix = dist_matrix.clamp(min=0)  # Numerical stability
    
    # Pred → Target: for each predicted point, find nearest GT point
    min_dist_p2t, _ = dist_matrix.min(dim=2)   # (B, N)
    cd_p2t = min_dist_p2t.mean(dim=1).mean()   # scalar
    
    if bidirectional:
        # Target → Pred: for each GT point, find nearest predicted point
        min_dist_t2p, _ = dist_matrix.min(dim=1)   # (B, M)
        cd_t2p = min_dist_t2p.mean(dim=1).mean()   # scalar
        cd_loss = cd_p2t + cd_t2p
    else:
        cd_t2p = torch.tensor(0.0, device=pred.device)
        cd_loss = cd_p2t
    
    return cd_loss, cd_p2t, cd_t2p


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance loss module."""
    
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cd_loss, _, _ = chamfer_distance(pred, target, self.bidirectional)
        return cd_loss


def f_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    Compute F-Score at a given distance threshold.
    
    F-Score = 2 * precision * recall / (precision + recall)
    
    Precision: fraction of predicted points within threshold of any GT point
    Recall: fraction of GT points within threshold of any predicted point
    
    Args:
        pred:      (B, N, 3) predicted point cloud
        target:    (B, M, 3) ground truth point cloud
        threshold: distance threshold
    
    Returns:
        fscore: (B,) F-Score per sample
    """
    # Pairwise distances
    pred_sq = (pred ** 2).sum(dim=2, keepdim=True)
    target_sq = (target ** 2).sum(dim=2, keepdim=True)
    dist_matrix = pred_sq + target_sq.transpose(1, 2) - 2 * torch.bmm(pred, target.transpose(1, 2))
    dist_matrix = dist_matrix.clamp(min=0).sqrt()  # Euclidean distances
    
    # Precision: for each predicted point, is its nearest GT point within threshold?
    min_dist_p2t, _ = dist_matrix.min(dim=2)  # (B, N)
    precision = (min_dist_p2t < threshold).float().mean(dim=1)  # (B,)
    
    # Recall: for each GT point, is its nearest predicted point within threshold?
    min_dist_t2p, _ = dist_matrix.min(dim=1)  # (B, M)
    recall = (min_dist_t2p < threshold).float().mean(dim=1)  # (B,)
    
    # F-Score
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    
    return fscore
# Aliases and missing functions

ChamferLoss = ChamferDistanceLoss

def evaluate_reconstruction(pred, gt, thresholds=[0.01, 0.02, 0.05]):
    with torch.no_grad():
        cd_loss, cd_p2g, cd_g2p = chamfer_distance(pred, gt, bidirectional=True)
        results = {
            'chamfer_distance': cd_loss.item(),
            'cd_pred_to_gt': cd_p2g.item(),
            'cd_gt_to_pred': cd_g2p.item(),
        }
        for t in thresholds:
            fs = f_score(pred, gt, threshold=t)
            results[f'f_score@{t}'] = fs.mean().item()
        return results

def dann_loss(domain_logits, domain_labels):
    return nn.functional.binary_cross_entropy_with_logits(
        domain_logits, domain_labels.float()
    )

def compute_dann_lambda(epoch, max_epoch, alpha=10.0):
    import math
    p = epoch / max_epoch
    return 2.0 / (1.0 + math.exp(-alpha * p)) - 1.0

if __name__ == "__main__":
    # Quick test
    B, N, M = 2, 2048, 2048
    pred = torch.randn(B, N, 3)
    target = torch.randn(B, M, 3)
    
    cd_loss, cd_p2t, cd_t2p = chamfer_distance(pred, target)
    print(f"Chamfer Distance: {cd_loss.item():.4f}")
    print(f"  P→T: {cd_p2t.item():.4f}")
    print(f"  T→P: {cd_t2p.item():.4f}")
    
    for thresh in [0.01, 0.02, 0.05]:
        fs = f_score(pred, target, threshold=thresh)
        print(f"F-Score@{thresh}: {fs.mean().item():.4f}")
