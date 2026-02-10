import torch
import sys
sys.path.append('.')
from model.make_model_promptsg import PromptSGModel

# Mock cfg
class MockCfg:
    def __init__(self):
        self.MODEL = type('model', (), {
            'NAME': 'ViT-B-16',
            'PROMPTSG': type('promptsg', (), {
                'CROSS_ATTN_HEADS': 8
            })()
        })()

cfg = MockCfg()

# Create model
model = PromptSGModel(num_classes=1501, camera_num=6, view_num=1, cfg=cfg)

print('=== TRAINABLE PARAMETERS ===')
total_trainable = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.shape} ({param.numel()} params)')
        total_trainable += param.numel()

print(f'\nTotal trainable parameters: {total_trainable:,}')

print('\n=== FROZEN PARAMETERS ===')
total_frozen = 0
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f'{name}: {param.shape} ({param.numel()} params)')
        total_frozen += param.numel()

print(f'\nTotal frozen parameters: {total_frozen:,}')
