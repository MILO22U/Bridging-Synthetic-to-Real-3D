"""Train Pix2Vox baseline for comparison."""
import argparse
import yaml
import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from pix2vox_baseline import Pix2VoxLite, Pix2VoxLoss, pointcloud_to_voxels, voxel_iou
from dataset import create_dataloaders
from losses import chamfer_distance, f_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = Pix2VoxLite(pretrained=True, voxel_size=32).to(device)
    criterion = Pix2VoxLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    save_dir = './checkpoints/pix2vox'
    os.makedirs(save_dir, exist_ok=True)

    best_cd = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            images, gt_points = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            with autocast():
                pred_voxels = model(images)
            loss = criterion(pred_voxels.float(), gt_points)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_cd_sum = 0
            val_f05_sum = 0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, gt_points = batch[0].to(device), batch[1].to(device)
                    pred_points = model.predict_points(images, n_points=cfg['data']['num_points'])
                    
                    for i in range(images.shape[0]):
                        cd = chamfer_distance(pred_points[i:i+1], gt_points[i:i+1])[0]
                        fs = f_score(pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), threshold=0.05)
                        val_cd_sum += cd.item()
                        val_f05_sum += fs.item()
                        n_val += 1

            val_cd = val_cd_sum / n_val
            val_f05 = val_f05_sum / n_val

            print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | Val CD: {val_cd:.6f} | F@0.05: {val_f05:.4f}")

            if val_cd < best_cd:
                best_cd = val_cd
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_cd': val_cd,
                    'val_f05': val_f05,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  ★ New best CD: {val_cd:.6f}")
        else:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f}")

    # Final test evaluation
    print("\n" + "="*60)
    print("Evaluating Pix2Vox on Test Set...")
    model.eval()
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_cd_sum = 0
    test_f01_sum = 0
    test_f02_sum = 0
    test_f05_sum = 0
    n_test = 0
    with torch.no_grad():
        for batch in test_loader:
            images, gt_points = batch[0].to(device), batch[1].to(device)
            pred_points = model.predict_points(images, n_points=cfg['data']['num_points'])
            
            for i in range(images.shape[0]):
                cd = chamfer_distance(pred_points[i:i+1], gt_points[i:i+1])[0]
                f01 = f_score(pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), threshold=0.01)
                f02 = f_score(pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), threshold=0.02)
                f05 = f_score(pred_points[i].unsqueeze(0), gt_points[i].unsqueeze(0), threshold=0.05)
                test_cd_sum += cd.item()
                test_f01_sum += f01.item()
                test_f02_sum += f02.item()
                test_f05_sum += f05.item()
                n_test += 1

    print(f"Pix2Vox Test Results (best epoch {ckpt['epoch']}):")
    print(f"  Chamfer Distance: {test_cd_sum/n_test:.6f}")
    print(f"  F-Score@0.01: {test_f01_sum/n_test:.4f}")
    print(f"  F-Score@0.02: {test_f02_sum/n_test:.4f}")
    print(f"  F-Score@0.05: {test_f05_sum/n_test:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
