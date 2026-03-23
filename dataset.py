"""
Dataset classes for Synthetic-to-Real 3D Reconstruction.
FIXED VERSION — Key changes:
  1. REMOVED RandomHorizontalFlip (was breaking image↔point cloud alignment)
  2. Cleaned up transform pipeline
"""

import os
import json
import glob
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────

class RandomBackground:
    """Paste a foreground object (with white/uniform background) onto a random real background."""
    
    def __init__(self, background_dir, threshold=240):
        self.background_dir = background_dir
        self.threshold = threshold
        self.bg_paths = []
        if background_dir and os.path.isdir(background_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.bg_paths.extend(glob.glob(os.path.join(background_dir, ext)))
            print(f"  RandomBackground: loaded {len(self.bg_paths)} backgrounds from {background_dir}")
        else:
            print(f"  WARNING: RandomBackground dir '{background_dir}' not found or empty!")
        
    def __call__(self, img):
        """img: PIL Image (RGB)"""
        if not self.bg_paths:
            return img
        
        # Load random background
        bg_path = random.choice(self.bg_paths)
        bg = Image.open(bg_path).convert('RGB').resize(img.size)
        
        # Create mask: assume white/near-white pixels are background
        img_arr = np.array(img)
        mask = np.all(img_arr > self.threshold, axis=-1)  # white background
        
        # Composite
        bg_arr = np.array(bg)
        result = img_arr.copy()
        result[mask] = bg_arr[mask]
        
        return Image.fromarray(result)


def get_train_transform(cfg_aug, use_random_bg=True):
    """
    Training transforms with augmentation (Strategy 1).
    
    FIX: Removed RandomHorizontalFlip — it was flipping images without
    flipping the corresponding GT point clouds, creating contradictory
    supervision signals that caused the model to predict symmetric blobs.
    """
    transforms_list = []
    
    # Random background replacement
    if use_random_bg and cfg_aug.get('use_random_backgrounds', False):
        transforms_list.append(RandomBackground(cfg_aug.get('background_dir', '')))
    
    transforms_list.extend([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=cfg_aug.get('random_crop_scale', (0.8, 1.0))),
        T.ColorJitter(
            brightness=cfg_aug.get('color_jitter', 0.4),
            contrast=cfg_aug.get('color_jitter', 0.4),
            saturation=cfg_aug.get('color_jitter', 0.4),
            hue=min(cfg_aug.get('color_jitter', 0.4) / 2, 0.5)
        ),
        # NOTE: RandomHorizontalFlip REMOVED — cannot flip images without also
        # flipping GT point cloud x-coordinates. This was a critical bug.
    ])
    
    if cfg_aug.get('gaussian_blur_prob', 0) > 0:
        transforms_list.append(
            T.RandomApply([T.GaussianBlur(kernel_size=cfg_aug['gaussian_blur_kernel'][1])],
                          p=cfg_aug['gaussian_blur_prob'])
        )
    
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=cfg_aug['normalize_mean'], std=cfg_aug['normalize_std']),
    ])
    
    if cfg_aug.get('random_erasing_prob', 0) > 0:
        transforms_list.append(T.RandomErasing(p=cfg_aug['random_erasing_prob']))
    
    return T.Compose(transforms_list)


def get_val_transform(cfg_aug):
    """Validation/test transforms (no augmentation)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=cfg_aug['normalize_mean'], std=cfg_aug['normalize_std']),
    ])


def get_tta_transforms(cfg_aug, n=10):
    """Test-time augmentation: returns a list of n slightly different transforms."""
    transforms_list = []
    for _ in range(n):
        t = T.Compose([
            T.Resize((224, 224)),
            T.RandomResizedCrop(224, scale=cfg_aug.get('tta_crop_scale', (0.85, 1.0))),
            T.ColorJitter(
                brightness=cfg_aug.get('tta_color_jitter', 0.2),
                contrast=cfg_aug.get('tta_color_jitter', 0.2),
                saturation=cfg_aug.get('tta_color_jitter', 0.2),
                hue=min(cfg_aug.get('tta_color_jitter', 0.2) / 2, 0.1)
            ),
            T.ToTensor(),
            T.Normalize(mean=cfg_aug['normalize_mean'], std=cfg_aug['normalize_std']),
        ])
        transforms_list.append(t)
    return transforms_list


# ──────────────────────────────────────────────────────────────
# ShapeNet + Cap3D Dataset (Synthetic Training)
# ──────────────────────────────────────────────────────────────

class ShapeNetCap3DDataset(Dataset):
    """
    Loads:
      - Rendered images from ShapeNet (or ShapeNet renders)
      - Point cloud ground truth from Cap3D (16,384 pts -> subsampled to N)
    
    Directory structure expected:
      shapenet_root/
        renders/
          <synset_id>/
            <model_id>/
              image_0000.png ... image_0023.png
      cap3d_root/
        point_clouds/
          <model_id>.npy       # (16384, 6) — xyz + rgb
    """
    
    def __init__(self, shapenet_root, cap3d_root, categories=None,
                 split='train', num_points=2048, num_views=24,
                 transform=None, split_ratio=(0.8, 0.1, 0.1)):
        super().__init__()
        self.shapenet_root = Path(shapenet_root)
        self.cap3d_root = Path(cap3d_root)
        self.num_points = num_points
        self.num_views = num_views
        self.transform = transform
        self.split = split
        
        # Collect all valid model IDs (have both renders + point cloud)
        self.samples = []
        render_dir = self.shapenet_root / "renders"
        pc_dir = self.cap3d_root / "point_clouds"
        
        if not render_dir.exists():
            print(f"Warning: Render directory {render_dir} not found. "
                  f"Run scripts/prepare_data.py first.")
            return
            
        synsets = categories or os.listdir(render_dir)
        for synset in synsets:
            synset_path = render_dir / synset
            if not synset_path.is_dir():
                continue
            for model_id in os.listdir(synset_path):
                model_path = synset_path / model_id
                pc_path = pc_dir / f"{model_id}.npy"
                if model_path.is_dir() and pc_path.exists():
                    self.samples.append({
                        'synset': synset,
                        'model_id': model_id,
                        'render_dir': str(model_path),
                        'pc_path': str(pc_path),
                    })
        
        # Sort for reproducibility, then split
        self.samples.sort(key=lambda x: x['model_id'])
        n = len(self.samples)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))
        
        if split == 'train':
            self.samples = self.samples[:train_end]
        elif split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif split == 'test':
            self.samples = self.samples[val_end:]
        
        print(f"ShapeNetCap3D [{split}]: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load random view
        view_idx = random.randint(0, self.num_views - 1)
        img_path = os.path.join(sample['render_dir'], f"image_{view_idx:04d}.png")
        
        # Fallback: try other naming conventions
        if not os.path.exists(img_path):
            imgs = sorted(glob.glob(os.path.join(sample['render_dir'], '*.png')))
            if imgs:
                img_path = random.choice(imgs)
            else:
                # Return a black image + zeros if missing
                img = Image.new('RGB', (224, 224), (0, 0, 0))
                pc = torch.zeros(self.num_points, 3)
                if self.transform:
                    img = self.transform(img)
                return img, pc, sample['model_id']
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Load point cloud
        pc = np.load(sample['pc_path'])  # (16384, 6) or (16384, 3)
        pc_xyz = pc[:, :3]  # take only xyz
        
        # Subsample to num_points
        if len(pc_xyz) > self.num_points:
            indices = np.random.choice(len(pc_xyz), self.num_points, replace=False)
            pc_xyz = pc_xyz[indices]
        elif len(pc_xyz) < self.num_points:
            # Pad by repeating
            indices = np.random.choice(len(pc_xyz), self.num_points, replace=True)
            pc_xyz = pc_xyz[indices]
        
        # Normalize point cloud to [-1, 1]
        centroid = pc_xyz.mean(axis=0)
        pc_xyz = pc_xyz - centroid
        max_dist = np.max(np.linalg.norm(pc_xyz, axis=1))
        if max_dist > 0:
            pc_xyz = pc_xyz / max_dist
        
        pc_tensor = torch.from_numpy(pc_xyz).float()
        
        return img, pc_tensor, sample['model_id']


# ──────────────────────────────────────────────────────────────
# Google Scanned Objects Dataset (Real-World Evaluation)
# ──────────────────────────────────────────────────────────────

class GSODataset(Dataset):
    """
    Google Scanned Objects for real-world evaluation.
    
    Directory structure:
      gso_root/
        <object_name>/
          renders/
            image_0000.png ...
          point_cloud.npy        # extracted from mesh
    """
    
    def __init__(self, gso_root, num_points=2048, transform=None):
        super().__init__()
        self.gso_root = Path(gso_root)
        self.num_points = num_points
        self.transform = transform
        self.samples = []
        
        if not self.gso_root.exists():
            print(f"Warning: GSO directory {gso_root} not found.")
            return
        
        for obj_dir in sorted(self.gso_root.iterdir()):
            if obj_dir.is_dir():
                renders = sorted(glob.glob(str(obj_dir / "renders" / "*.png")))
                pc_path = obj_dir / "point_cloud.npy"
                if renders and pc_path.exists():
                    for render in renders:
                        self.samples.append({
                            'image_path': render,
                            'pc_path': str(pc_path),
                            'object_name': obj_dir.name,
                        })
        
        print(f"GSO Dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Load point cloud
        pc = np.load(sample['pc_path'])[:, :3]
        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
            pc = pc[indices]
        
        # Normalize
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.linalg.norm(pc, axis=1))
        if max_dist > 0:
            pc = pc / max_dist
        
        pc_tensor = torch.from_numpy(pc).float()
        
        return img, pc_tensor, sample['object_name']


# ──────────────────────────────────────────────────────────────
# Real Photo Dataset (Qualitative Evaluation)
# ──────────────────────────────────────────────────────────────

class RealPhotoDataset(Dataset):
    """Load user-captured phone photos for qualitative evaluation."""
    
    def __init__(self, image_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.image_paths = []
        
        if os.path.isdir(image_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, ext))))
        
        print(f"Real Photos: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        name = os.path.splitext(os.path.basename(img_path))[0]
        return img, name


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────

def create_dataloaders(cfg):
    """Create train/val/test dataloaders from config dict."""
    aug_cfg = cfg['augmentation']
    data_cfg = cfg['data']
    train_cfg = cfg['training']
    
    train_transform = get_train_transform(aug_cfg)
    val_transform = get_val_transform(aug_cfg)
    
    train_dataset = ShapeNetCap3DDataset(
        shapenet_root=data_cfg['shapenet_root'],
        cap3d_root=data_cfg['cap3d_root'],
        categories=data_cfg.get('train_categories'),
        split='train',
        num_points=data_cfg['num_points'],
        num_views=data_cfg['num_views'],
        transform=train_transform,
    )
    
    val_dataset = ShapeNetCap3DDataset(
        shapenet_root=data_cfg['shapenet_root'],
        cap3d_root=data_cfg['cap3d_root'],
        categories=data_cfg.get('train_categories'),
        split='val',
        num_points=data_cfg['num_points'],
        num_views=data_cfg['num_views'],
        transform=val_transform,
    )
    
    test_dataset = ShapeNetCap3DDataset(
        shapenet_root=data_cfg['shapenet_root'],
        cap3d_root=data_cfg['cap3d_root'],
        categories=data_cfg.get('train_categories'),
        split='test',
        num_points=data_cfg['num_points'],
        num_views=data_cfg['num_views'],
        transform=val_transform,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'],
        shuffle=True, num_workers=train_cfg['num_workers'],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg['batch_size'],
        shuffle=False, num_workers=train_cfg['num_workers'],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg['batch_size'],
        shuffle=False, num_workers=train_cfg['num_workers'],
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
