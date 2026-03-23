import torch
ckpt = torch.load('checkpoints/base_pretrained/best_model.pt', map_location='cpu', weights_only=False)
if 'config' in ckpt:
    import json
    print(json.dumps(ckpt['config'], indent=2))
else:
    # Check model state dict for shapes
    for k, v in ckpt['model_state_dict'].items():
        if 'query' in k or 'point_head' in k:
            print(f'{k}: {v.shape}')
