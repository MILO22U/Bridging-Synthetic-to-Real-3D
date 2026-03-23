"""Quick real photo inference — check if model produces blobs or shapes."""
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

def remove_background(img_pil):
    try:
        from rembg import remove
        result = remove(img_pil)
        bg = Image.new('RGB', result.size, (0, 0, 0))
        bg.paste(result, mask=result.split()[3])
        return bg
    except ImportError:
        return img_pil

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    cfg['model']['num_query_tokens'] = 1024
    model = HybridReconstructor(cfg['model']).to(device)
    ckpt = torch.load('checkpoints/base_pretrained/best_model.pt',
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    photo_dir = './data/real_photos'
    save_dir = './visualizations/real_inference_check'
    os.makedirs(save_dir, exist_ok=True)

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    photos = sorted([f for f in os.listdir(photo_dir)
                     if os.path.splitext(f)[1].lower() in valid_ext])[:6]

    for idx, fname in enumerate(photos):
        img = Image.open(os.path.join(photo_dir, fname)).convert('RGB')
        img_nobg = remove_background(img)

        with torch.no_grad():
            pred_orig = model(transform(img).unsqueeze(0).to(device))[0].cpu().numpy()
            pred_nobg = model(transform(img_nobg).unsqueeze(0).to(device))[0].cpu().numpy()

        fig = plt.figure(figsize=(16, 4))

        ax1 = fig.add_subplot(141)
        ax1.imshow(img.resize((224, 224)))
        ax1.set_title('Original'); ax1.axis('off')

        ax2 = fig.add_subplot(142)
        ax2.imshow(img_nobg.resize((224, 224)))
        ax2.set_title('BG Removed'); ax2.axis('off')

        ax3 = fig.add_subplot(143, projection='3d')
        ax3.scatter(pred_orig[:, 0], pred_orig[:, 1], pred_orig[:, 2], s=0.5, alpha=0.5)
        ax3.set_title('Pred (Original)')

        ax4 = fig.add_subplot(144, projection='3d')
        ax4.scatter(pred_nobg[:, 0], pred_nobg[:, 1], pred_nobg[:, 2], s=0.5, alpha=0.5, c='coral')
        ax4.set_title('Pred (BG Removed)')

        name = os.path.splitext(fname)[0]
        plt.suptitle(name[:40], fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'real_{idx}.png'), dpi=150)
        plt.close()

        # Print spread stats
        for label, pts in [('orig', pred_orig), ('nobg', pred_nobg)]:
            spread = pts.std(axis=0).mean()
            extent = pts.max(axis=0) - pts.min(axis=0)
            print(f"  {fname[:30]:30s} [{label}] spread={spread:.4f} extent={extent}")

    print(f"\nSaved to {save_dir}/")

if __name__ == '__main__':
    main()
