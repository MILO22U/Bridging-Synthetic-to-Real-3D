"""Run visualization on trained model."""
import argparse
import yaml
import torch
from model import HybridReconstructor
from dataset import create_dataloaders
from visualize import generate_visualizations, plot_training_curves
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--save_dir', default='./visualizations')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = HybridReconstructor(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load test data
    _, _, test_loader = create_dataloaders(cfg)
    
    # Generate visualizations
    print(f"Generating {args.n_samples} visualizations...")
    generate_visualizations(model, test_loader, device, args.save_dir, n_samples=args.n_samples)

    # Plot training curves if log exists
    log_file = os.path.join(os.path.dirname(args.checkpoint), 'training_log.json')
    if os.path.exists(log_file):
        plot_training_curves(log_file, save_path=os.path.join(args.save_dir, 'training_curves.png'))
        print("Saved training curves")

    print(f"\nDone! Check {args.save_dir}/")

if __name__ == '__main__':
    main()
