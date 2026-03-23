"""Quick VRAM test with different batch sizes and 2048 points."""
import torch
from model import HybridReconstructor

device = torch.device('cuda')
cfg_model = {
    'encoder_backbone': 'resnet18', 'use_pretrained': True,
    'encoder_dim': 512, 'num_image_tokens': 49,
    'num_query_tokens': 2048, 'query_dim': 256,
    'cross_attn_layers': 2, 'cross_attn_heads': 8,
    'self_attn_layers': 4, 'self_attn_heads': 8,
    'mlp_hidden_dim': 512, 'dropout': 0.1, 'output_dim': 3,
}

model = HybridReconstructor(cfg_model).to(device)

for bs in [16, 20, 24, 28, 32]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(bs, 3, 224, 224, device=device)
        with torch.amp.autocast('cuda'):
            pred = model(x)
            loss = pred.sum()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"batch_size={bs}: peak VRAM = {peak:.2f} GB  ({'OK' if peak < 15 else 'TIGHT'})")
        del x, pred, loss
    except RuntimeError as e:
        print(f"batch_size={bs}: OOM!")
        break

model.zero_grad()
torch.cuda.empty_cache()
