"""Shared helpers for the scope study.  Independent of scripts/nslib.py.

Only `rtlib` (the cross-check's literal float64 estimator) is reused, as instructed.
"""
import os, sys, json, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
VAL = os.path.join(ROOT, 'data', 'in1k', 'val')
OUT = os.path.join(ROOT, 'outputs', 'scope')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'crosscheck'))


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print(f'  -> {p}', flush=True)
    return p


def tfm(cfg):
    from torchvision import transforms
    isz = cfg['input_size'][-1]
    ip = {'bicubic': transforms.InterpolationMode.BICUBIC,
          'bilinear': transforms.InterpolationMode.BILINEAR,
          'nearest': transforms.InterpolationMode.NEAREST}[cfg.get('interpolation', 'bicubic')]
    return transforms.Compose([
        transforms.Resize(int(isz / cfg.get('crop_pct', 0.9)), interpolation=ip),
        transforms.CenterCrop(isz), transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std'])])


def val_ds(cfg):
    from torchvision import datasets
    return datasets.ImageFolder(VAL, tfm(cfg))


def val_images(cfg, idx):
    ds = val_ds(cfg)
    return torch.stack([ds[i][0] for i in idx])


# ------------------------------------------------------------------ affinity
@torch.no_grad()
def seg_dev(fwd, fa, fb, LAM=(0.0, .07, .13, .25, .5, .61, .75, .9, 1.0)):
    """max_lam |z(lam) - lerp| / |z(1)-z(0)| for z(lam)=fwd(fa+lam(fb-fa)).

    0  <=>  fwd is affine on the segment  <=>  the readout is additive over any
    partition of the cell displacement (the stage-4 criterion).
    """
    z = torch.stack([fwd(fa + l * (fb - fa))[0] for l in LAM])
    lam = torch.tensor(LAM, device=z.device, dtype=z.dtype)[:, None]
    lerp = z[0][None] + lam * (z[-1] - z[0])[None]
    sc = (z[-1] - z[0]).abs().max()
    return float((z - lerp).abs().max() / (sc + 1e-300)), float(sc)


# ------------------------------------------------------ module surgery utils
NORMISH = ('LayerNorm', 'LayerNorm2d', 'LayerNormExp2d', 'RmsNorm', 'RmsNorm2d',
           'GroupNorm', 'InstanceNorm2d', 'LayerNormFp32', 'RmsNormFp32')
ACTISH = ('ReLU', 'ReLU6', 'GELU', 'QuickGELU', 'SiLU', 'Swish', 'Hardswish',
          'HardSwish', 'Hardsigmoid', 'HardSigmoid', 'Tanh', 'Sigmoid', 'Mish',
          'ELU', 'LeakyReLU', 'PReLU', 'SquaredReLU', 'StarReLU', 'GELUTanh',
          'Hardtanh', 'SELU', 'CELU', 'Softplus')


def is_norm(mod):
    n = type(mod).__name__
    return n in NORMISH or (n.endswith('Norm') and not n.startswith('Batch')) \
        or n.startswith('RmsNorm') or n.startswith('LayerNorm')


def is_act(mod):
    return type(mod).__name__ in ACTISH


def swap(root, pred, make, path=''):
    """Replace every submodule satisfying pred(mod) with make(mod). Returns names."""
    hit = []
    for name, ch in list(root.named_children()):
        p = f'{path}.{name}' if path else name
        if pred(ch):
            setattr(root, name, make(ch)); hit.append(f'{p}:{type(ch).__name__}')
        else:
            hit += swap(ch, pred, make, p)
    return hit
