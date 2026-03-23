"""
Pix2Vox-Lite Baseline for comparison.

A simplified version of Pix2Vox (Xie et al., ICCV 2019) that predicts
voxel grids instead of point clouds. Used as a baseline to compare
domain gap across different 3D representation types.

Reference: "Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class Pix2VoxEncoder(nn.Module):
    """ResNet-18 encoder (same backbone as our model for fair comparison)."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1024)
    
    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        feat = self.fc(feat)
        return feat


class Pix2VoxDecoder(nn.Module):
    """3D CNN decoder: generates a voxel grid from image features."""
    
    def __init__(self, input_dim=1024, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size
        
        # Project to 3D feature volume
        self.fc = nn.Linear(input_dim, 256 * 2 * 2 * 2)
        
        self.decoder = nn.Sequential(
            # 2x2x2 -> 4x4x4
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 2, 2, 2)
        x = self.decoder(x)
        return x.squeeze(1)  # (B, 32, 32, 32)


class Pix2VoxLite(nn.Module):
    """
    Simplified Pix2Vox for single-image voxel reconstruction.
    
    Input: (B, 3, 224, 224) image
    Output: (B, V, V, V) voxel occupancy grid
    """
    
    def __init__(self, pretrained=True, voxel_size=32):
        super().__init__()
        self.encoder = Pix2VoxEncoder(pretrained=pretrained)
        self.decoder = Pix2VoxDecoder(voxel_size=voxel_size)
        self.voxel_size = voxel_size
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Pix2VoxLite: {n_params / 1e6:.1f}M parameters, "
              f"voxel size: {voxel_size}^3")
    
    def forward(self, x):
        feat = self.encoder(x)
        voxels = self.decoder(feat)
        return voxels
    
    def predict_points(self, x, threshold=0.3, n_points=2048):
        """
        Predict point cloud from voxels for fair comparison with our model.
        
        1. Predict voxel grid
        2. Threshold to get occupied voxels
        3. Convert voxel centers to point cloud
        4. Subsample to n_points
        """
        voxels = self.forward(x)  # (B, V, V, V)
        B = voxels.shape[0]
        
        all_points = []
        for b in range(B):
            # Get occupied voxel indices
            occupied = (voxels[b] > threshold).nonzero(as_tuple=False).float()
            
            if len(occupied) == 0:
                # Fallback: return grid center points
                points = torch.zeros(n_points, 3, device=x.device)
            else:
                # Normalize to [-1, 1]
                points = (occupied / self.voxel_size) * 2 - 1
                
                # Subsample
                if len(points) > n_points:
                    indices = torch.randperm(len(points))[:n_points]
                    points = points[indices]
                elif len(points) < n_points:
                    indices = torch.randint(0, len(points), (n_points,))
                    points = points[indices]
            
            all_points.append(points)
        
        return torch.stack(all_points)  # (B, n_points, 3)


def voxel_iou(pred_voxels, gt_voxels, threshold=0.3):
    """
    Intersection over Union for voxel grids.
    
    Args:
        pred_voxels: (B, V, V, V) predicted occupancy
        gt_voxels: (B, V, V, V) ground truth occupancy
        threshold: binarization threshold
    """
    pred_binary = (pred_voxels > threshold).float()
    gt_binary = (gt_voxels > threshold).float()
    
    intersection = (pred_binary * gt_binary).sum(dim=[1, 2, 3])
    union = ((pred_binary + gt_binary) > 0).float().sum(dim=[1, 2, 3])
    
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


# ──────────────────────────────────────────────────────────────
# Training utilities for Pix2Vox baseline
# ──────────────────────────────────────────────────────────────

def pointcloud_to_voxels(points, voxel_size=32):
    """
    Convert point cloud to voxel grid.
    
    Args:
        points: (B, N, 3) in [-1, 1]
        voxel_size: grid resolution
    Returns:
        (B, V, V, V) binary voxel grid
    """
    B, N, _ = points.shape
    
    # Map [-1, 1] to [0, voxel_size-1]
    indices = ((points + 1) / 2 * (voxel_size - 1)).long()
    indices = indices.clamp(0, voxel_size - 1)
    
    voxels = torch.zeros(B, voxel_size, voxel_size, voxel_size,
                         device=points.device)
    
    for b in range(B):
        voxels[b, indices[b, :, 0], indices[b, :, 1], indices[b, :, 2]] = 1.0
    
    return voxels


class Pix2VoxLoss(nn.Module):
    """Binary cross-entropy loss for voxel prediction."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_voxels, gt_points, voxel_size=32):
        gt_voxels = pointcloud_to_voxels(gt_points, voxel_size)
        return F.binary_cross_entropy(pred_voxels, gt_voxels)
