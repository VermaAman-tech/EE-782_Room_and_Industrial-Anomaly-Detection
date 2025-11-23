import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg.get('device') == 'auto':
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

