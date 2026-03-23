"""
Real Photo Inference with Background Removal

The model was trained on black-background synthetic renders.
Real photos have complex backgrounds that confuse the model.
Solution: Remove background first, then run inference.
"""
import argparse
import yaml
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import HybridReconstructor

def remove_background(img_pil):
    """Remove background using rembg, replace with black."""
    try:
        from rembg import remove
        # Remove background (returns RGBA)
        result = remove(img_pil)
        # Convert RGBA to RGB with black background
        bg = Image.new('RGB', result.size, (0, 0, 0))
        bg.paste(result, mask=result.split()[3])
        return bg
    except ImportError:
        print("  rembg not installed, using simple threshold")
        return simple_bg_removal(img_pil)

def simple_bg_removal(img_pil):
    """Simple background removal: darken non-central regions."""
    img = np.array(img_pil).astype(float)
    h, w = img.shape[:2]
    
    # Create a soft mask: center is bright, edges are dark
    Y, X = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    mask = np.exp(-((X-cx)**2/(w*0.3)**2 + (Y-cy)**2/(h*0.3)**2))
    mask = mask[..., np.newaxis]
    
    # Apply mask
    img = img * (0.3 + 0.7 * mask)
    return Image.fromarray(img.astype(np.uint8))

def plot_point_cloud(ax, points, color='steelblue', title=''):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=0.5, alpha=0.6)
    ax.set_title(title, fontsize=9)
    max_range = max(np.max(np.abs(points)) * 1.1, 0.5)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.view_init(elev=30, azim=45)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--photo_dir', default='./data/real_photos')
    parser.add_argument('--save_dir', default='./visualizations/real_inference')
    parser.add_argument('--no_rembg', action='store_true', help='Skip rembg, use simple method')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print("=" * 60)
    print("Real Photo Inference with Background Removal")
    print("=" * 60)
    model = HybridReconstructor(cfg['model']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_vis = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # Process each photo
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    photos = sorted([f for f in os.listdir(args.photo_dir) 
                     if os.path.splitext(f)[1].lower() in valid_ext])
    
    print(f"Processing {len(photos)} photos...\n")

    for idx, fname in enumerate(photos):
        name = os.path.splitext(fname)[0]
        img_path = os.path.join(args.photo_dir, fname)
        
        try:
            img_orig = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue
        
        # Remove background
        if args.no_rembg:
            img_nobg = simple_bg_removal(img_orig)
        else:
            img_nobg = remove_background(img_orig)
        
        # Prepare tensors
        orig_tensor = transform(img_orig).unsqueeze(0).to(device)
        nobg_tensor = transform(img_nobg).unsqueeze(0).to(device)
        
        orig_vis = transform_vis(img_orig)
        nobg_vis = transform_vis(img_nobg)
        
        # Inference
        with torch.no_grad():
            pred_orig = model(orig_tensor)[0]
            pred_nobg = model(nobg_tensor)[0]
        
        # Visualization: 4 panels
        fig = plt.figure(figsize=(20, 5))
        
        # Original photo
        ax1 = fig.add_subplot(141)
        ax1.imshow(orig_vis.permute(1, 2, 0).numpy())
        ax1.set_title('Original Photo', fontsize=10)
        ax1.axis('off')
        
        # Background removed
        ax2 = fig.add_subplot(142)
        ax2.imshow(nobg_vis.permute(1, 2, 0).numpy())
        ax2.set_title('Background Removed', fontsize=10)
        ax2.axis('off')
        
        # Pred from original
        ax3 = fig.add_subplot(143, projection='3d')
        plot_point_cloud(ax3, pred_orig, 'steelblue', 'Pred (Original)')
        
        # Pred from bg-removed
        ax4 = fig.add_subplot(144, projection='3d')
        plot_point_cloud(ax4, pred_nobg, 'coral', 'Pred (BG Removed)')
        
        plt.suptitle(name, fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f'real_{name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [{idx+1}/{len(photos)}] {name}")

    print(f"\nDone! Saved to {args.save_dir}/")

if __name__ == '__main__':
    main()
