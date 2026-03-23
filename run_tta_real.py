"""Run TTA inference on real photos — compare no-TTA vs TTA quality."""
import torch
import yaml
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import HybridReconstructor
from strategies import TestTimeAugmentation

def remove_background(img_pil):
    try:
        from rembg import remove
        result = remove(img_pil)
        bg = Image.new('RGB', result.size, (0, 0, 0))
        bg.paste(result, mask=result.split()[3])
        return bg
    except ImportError:
        print("  rembg not installed, using original image")
        return img_pil

def plot_pc(ax, points, color='steelblue', title=''):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=0.3, alpha=0.5)
    ax.set_title(title, fontsize=8)
    r = max(np.max(np.abs(points)) * 1.1, 0.5)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.view_init(elev=25, azim=45)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    cfg['model']['num_query_tokens'] = 2048
    model = HybridReconstructor(cfg['model']).to(device)
    ckpt = torch.load('checkpoints/retrain_2048/best.pt',
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}")

    # TTA with 10 augmentations
    tta = TestTimeAugmentation(num_augments=10, image_size=224)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    photo_dir = './data/real_photos'
    save_dir = './visualizations2/tta_real'
    os.makedirs(save_dir, exist_ok=True)

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    photos = sorted([f for f in os.listdir(photo_dir)
                     if os.path.splitext(f)[1].lower() in valid_ext])

    print(f"Processing {len(photos)} photos with TTA...\n")

    for idx, fname in enumerate(photos):
        name = os.path.splitext(fname)[0][:40]
        img = Image.open(os.path.join(photo_dir, fname)).convert('RGB')
        img_nobg = remove_background(img)

        nobg_tensor = transform(img_nobg).unsqueeze(0).to(device)

        with torch.no_grad():
            # Without TTA
            pred_plain = model(nobg_tensor)[0]
            # With TTA
            pred_tta = tta(model, nobg_tensor)[0]
        torch.cuda.empty_cache()

        fig = plt.figure(figsize=(16, 4))

        # Original
        ax1 = fig.add_subplot(141)
        ax1.imshow(img.resize((224, 224)))
        ax1.set_title('Original', fontsize=9); ax1.axis('off')

        # BG removed
        ax2 = fig.add_subplot(142)
        ax2.imshow(img_nobg.resize((224, 224)))
        ax2.set_title('BG Removed', fontsize=9); ax2.axis('off')

        # Plain prediction
        ax3 = fig.add_subplot(143, projection='3d')
        plot_pc(ax3, pred_plain, 'steelblue', 'No TTA')

        # TTA prediction
        ax4 = fig.add_subplot(144, projection='3d')
        plot_pc(ax4, pred_tta, 'coral', 'With TTA (10 aug)')

        plt.suptitle(name, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'tta_{idx:02d}_{name}.png'), dpi=150)
        plt.close()

        p1 = pred_plain.cpu().numpy()
        p2 = pred_tta.cpu().numpy()
        print(f"  [{idx+1}/{len(photos)}] {name}")
        print(f"    Plain: spread={p1.std(axis=0).mean():.4f}")
        print(f"    TTA:   spread={p2.std(axis=0).mean():.4f}")

    print(f"\nDone! Results saved to {save_dir}/")

if __name__ == '__main__':
    main()
