"""
Configuration for Synthetic-to-Real 3D Reconstruction Project
AI 535 — Deep Learning

Supports two modes:
  1. Dict-based from config.yaml (for model.py / dataset.py)
  2. Attribute-based dataclass (for train.py / evaluate.py)

Usage:
    cfg = load_config('config.yaml')       # dict mode
    cfg = get_config('base')               # dataclass mode
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YAML-based config loader (used by model.py, dataset.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataclass-based config (used by train.py, evaluate.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DataConfig:
    shapenet_root: str = "./data/shapenet"
    cap3d_root: str = "./data/cap3d"
    gso_root: str = "./data/gso"
    real_photos_dir: str = "./data/real_photos"
    categories: List[str] = field(default_factory=lambda: [
        "02691156", "02828884", "02933112", "02958343", "03001627",
        "03211117", "03636649", "03691459", "04090263", "04256520",
        "04379243", "04401088", "04530566",
    ])
    image_size: int = 224
    num_points: int = 2048
    num_views: int = 24
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    encoder_backbone: str = "resnet18"
    use_pretrained: bool = True         # ← Professor feedback: USE pretrained!
    encoder_dim: int = 512
    num_image_tokens: int = 49
    num_query_tokens: int = 2048
    query_dim: int = 256
    cross_attn_layers: int = 2
    cross_attn_heads: int = 8
    self_attn_layers: int = 6
    self_attn_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout: float = 0.1
    output_dim: int = 3


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    encoder_lr: float = 1e-4            # Lower LR for pretrained encoder
    decoder_lr: float = 5e-4            # Higher LR for random-init decoder
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    use_amp: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    log_dir: str = "./outputs/logs"
    log_every: int = 50
    patience: int = 30
    seed: int = 42


@dataclass
class AugmentationConfig:
    use_random_backgrounds: bool = True
    background_dir: str = "./data/backgrounds"
    color_jitter: float = 0.4
    gaussian_blur_prob: float = 0.3
    gaussian_blur_kernel: List[int] = field(default_factory=lambda: [5, 9])
    random_crop_scale: List[float] = field(default_factory=lambda: [0.8, 1.0])
    random_erasing_prob: float = 0.1
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    tta_num_augmentations: int = 10
    tta_color_jitter: float = 0.2
    tta_crop_scale: List[float] = field(default_factory=lambda: [0.85, 1.0])


@dataclass
class DomainAdaptConfig:
    dann_enabled: bool = False
    dann_lambda: float = 1.0
    dann_alpha: float = 10.0
    dann_discriminator_hidden: int = 256
    dann_lr: float = 1e-4
    adain_enabled: bool = False
    adain_style_weight: float = 1.0
    adain_content_weight: float = 1.0


@dataclass
class EvalConfig:
    fscore_thresholds: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    num_vis_samples: int = 20
    vis_output_dir: str = "./visualizations"


@dataclass
class Config:
    """Master configuration — attribute-based access."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    domain_adapt: DomainAdaptConfig = field(default_factory=DomainAdaptConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cuda"
    seed: int = 42
    experiment_name: str = "base_pretrained"

    def to_dict(self) -> dict:
        """Convert to dict format compatible with model.py / dataset.py."""
        return {
            'model': {
                'image_size': self.data.image_size,
                'encoder_backbone': self.model.encoder_backbone,
                'use_pretrained': self.model.use_pretrained,
                'encoder_dim': self.model.encoder_dim,
                'num_image_tokens': self.model.num_image_tokens,
                'num_query_tokens': self.model.num_query_tokens,
                'query_dim': self.model.query_dim,
                'cross_attn_layers': self.model.cross_attn_layers,
                'cross_attn_heads': self.model.cross_attn_heads,
                'self_attn_layers': self.model.self_attn_layers,
                'self_attn_heads': self.model.self_attn_heads,
                'mlp_hidden_dim': self.model.mlp_hidden_dim,
                'dropout': self.model.dropout,
                'output_dim': self.model.output_dim,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'scheduler': self.training.scheduler,
                'warmup_epochs': self.training.warmup_epochs,
                'grad_clip': self.training.grad_clip,
                'num_workers': self.data.num_workers,
                'seed': self.training.seed,
                'mixed_precision': self.training.use_amp,
            },
            'data': {
                'shapenet_root': self.data.shapenet_root,
                'cap3d_root': self.data.cap3d_root,
                'gso_root': self.data.gso_root,
                'real_images_dir': self.data.real_photos_dir,
                'num_views': self.data.num_views,
                'num_points': self.data.num_points,
                'train_categories': self.data.categories,
                'train_split': self.data.train_ratio,
                'val_split': self.data.val_ratio,
                'test_split': self.data.test_ratio,
            },
            'augmentation': {
                'use_random_backgrounds': self.augmentation.use_random_backgrounds,
                'background_dir': self.augmentation.background_dir,
                'color_jitter': self.augmentation.color_jitter,
                'gaussian_blur_prob': self.augmentation.gaussian_blur_prob,
                'gaussian_blur_kernel': self.augmentation.gaussian_blur_kernel,
                'random_crop_scale': self.augmentation.random_crop_scale,
                'random_erasing_prob': self.augmentation.random_erasing_prob,
                'normalize_mean': self.augmentation.normalize_mean,
                'normalize_std': self.augmentation.normalize_std,
                'tta_num_augmentations': self.augmentation.tta_num_augmentations,
                'tta_color_jitter': self.augmentation.tta_color_jitter,
                'tta_crop_scale': self.augmentation.tta_crop_scale,
            },
            'domain_adaptation': {
                'dann_enabled': self.domain_adapt.dann_enabled,
                'dann_lambda': self.domain_adapt.dann_lambda,
                'dann_alpha': self.domain_adapt.dann_alpha,
                'dann_discriminator_hidden': self.domain_adapt.dann_discriminator_hidden,
                'adain_enabled': self.domain_adapt.adain_enabled,
            },
            'evaluation': {
                'fscore_thresholds': self.eval.fscore_thresholds,
                'num_visualization_samples': self.eval.num_vis_samples,
            },
        }


def get_config(experiment: str = "base") -> Config:
    """Get pre-defined experiment configuration."""
    cfg = Config()

    if experiment == "base":
        cfg.experiment_name = "base_pretrained"
        cfg.model.use_pretrained = True

    elif experiment == "scratch":
        cfg.experiment_name = "scratch_resnet18"
        cfg.model.use_pretrained = False
        cfg.training.encoder_lr = 5e-4
        cfg.training.num_epochs = 150

    elif experiment == "augmented":
        cfg.experiment_name = "augmented_training"
        cfg.augmentation.use_random_backgrounds = True
        cfg.augmentation.color_jitter = 0.5
        cfg.augmentation.gaussian_blur_prob = 0.4
        cfg.augmentation.random_erasing_prob = 0.2

    elif experiment == "dann":
        cfg.experiment_name = "dann_adaptation"
        cfg.domain_adapt.dann_enabled = True

    elif experiment == "full":
        cfg.experiment_name = "full_adaptation"
        cfg.augmentation.use_random_backgrounds = True
        cfg.augmentation.color_jitter = 0.5
        cfg.domain_adapt.dann_enabled = True

    return cfg


def config_from_yaml(yaml_path: str) -> Config:
    """Create a Config dataclass from a YAML file."""
    raw = load_config(yaml_path)
    cfg = Config()

    # Model
    m = raw.get('model', {})
    cfg.model.encoder_backbone = m.get('encoder_backbone', cfg.model.encoder_backbone)
    cfg.model.use_pretrained = m.get('use_pretrained', cfg.model.use_pretrained)
    cfg.model.encoder_dim = m.get('encoder_dim', cfg.model.encoder_dim)
    cfg.model.num_image_tokens = m.get('num_image_tokens', cfg.model.num_image_tokens)
    cfg.model.num_query_tokens = m.get('num_query_tokens', cfg.model.num_query_tokens)
    cfg.model.query_dim = m.get('query_dim', cfg.model.query_dim)
    cfg.model.cross_attn_layers = m.get('cross_attn_layers', cfg.model.cross_attn_layers)
    cfg.model.cross_attn_heads = m.get('cross_attn_heads', cfg.model.cross_attn_heads)
    cfg.model.self_attn_layers = m.get('self_attn_layers', cfg.model.self_attn_layers)
    cfg.model.self_attn_heads = m.get('self_attn_heads', cfg.model.self_attn_heads)
    cfg.model.mlp_hidden_dim = m.get('mlp_hidden_dim', cfg.model.mlp_hidden_dim)
    cfg.model.dropout = m.get('dropout', cfg.model.dropout)
    cfg.model.output_dim = m.get('output_dim', cfg.model.output_dim)

    # Training
    t = raw.get('training', {})
    cfg.training.batch_size = t.get('batch_size', cfg.training.batch_size)
    cfg.training.num_epochs = t.get('num_epochs', cfg.training.num_epochs)
    cfg.training.learning_rate = t.get('learning_rate', cfg.training.learning_rate)
    cfg.training.weight_decay = t.get('weight_decay', cfg.training.weight_decay)
    cfg.training.scheduler = t.get('scheduler', cfg.training.scheduler)
    cfg.training.warmup_epochs = t.get('warmup_epochs', cfg.training.warmup_epochs)
    cfg.training.grad_clip = t.get('grad_clip', cfg.training.grad_clip)
    cfg.training.use_amp = t.get('mixed_precision', cfg.training.use_amp)
    cfg.training.seed = t.get('seed', cfg.training.seed)

    # Data
    d = raw.get('data', {})
    cfg.data.shapenet_root = d.get('shapenet_root', cfg.data.shapenet_root)
    cfg.data.cap3d_root = d.get('cap3d_root', cfg.data.cap3d_root)
    cfg.data.gso_root = d.get('gso_root', cfg.data.gso_root)
    cfg.data.real_photos_dir = d.get('real_images_dir', cfg.data.real_photos_dir)
    cfg.data.num_views = d.get('num_views', cfg.data.num_views)
    cfg.data.num_points = d.get('num_points', cfg.data.num_points)
    if 'train_categories' in d:
        cfg.data.categories = d['train_categories']

    # Augmentation
    a = raw.get('augmentation', {})
    cfg.augmentation.use_random_backgrounds = a.get('use_random_backgrounds', cfg.augmentation.use_random_backgrounds)
    cfg.augmentation.background_dir = a.get('background_dir', cfg.augmentation.background_dir)
    cfg.augmentation.color_jitter = a.get('color_jitter', cfg.augmentation.color_jitter)

    return cfg
