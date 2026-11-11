"""K2 shared machinery: end-to-end fine-tuned ResNet-50 with four heads on CUB-200-2011.

Protocol: protocols/K2_end_to_end_protocol.md (read only).

Reused project code (imported, not re-implemented):
  scripts/heads.py          GapFC (H0), LSE T=1 (H1), Attn = gated attention of Ilse et al. (H2)
  scripts/wsol/wlib.py      cub_manifest, ns_gap, ns_lse, ns_pooled, upsample, box_curve,
                            maxboxacc_v2, holm, PROTOCOL (bank size, thresholds, deltas, eval res)
  scripts/crosscheck/rtlib.py  NS_literal (literal float64 substitution loop) for the checks
H3 = analyses/theory_sweeps/scripts/theory_sweeps_lib.py:LNPost (fc(LN(mean_t a_t))); copied here verbatim
because importing theory_sweeps_lib changes global cuDNN TF32 flags and loads cached Imagenette features.
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
K2 = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(K2, '..', '..'))
OUT = os.path.join(K2, 'out')
LOGS = os.path.join(OUT, 'logs')
CKPT = os.path.join(K2, 'checkpoints')
for d in (OUT, LOGS, CKPT, os.path.join(OUT, 'per_image')):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'wsol'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'crosscheck'))
import heads as HD   # noqa: E402
import wlib as W     # noqa: E402

HEADS = ['H0', 'H1', 'H2', 'H3']
SEEDS = [0, 1, 2]
HEAD_NAME = {'H0': 'gap_fc', 'H1': 'lse1.0', 'H2': 'attn', 'H3': 'ln_post'}
HEAD_DESC = {'H0': 'global average pooling + linear (degenerate)',
             'H1': 'log-sum-exp pooling of per-cell class scores, T = +1',
             'H2': 'gated attention pooling (Ilse et al.) + linear',
             'H3': 'global average pooling + LayerNorm + linear'}
K = 200
C = 2048
SIZE = 448
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # torchvision IMAGENET1K_V1

RECIPE = dict(optimizer='SGD', momentum=0.9, weight_decay=1e-4, lr_head=0.01, lr_backbone=0.001,
              schedule='cosine (per iteration, no warm-up)', epochs=30, batch=32, amp='float16',
              train_tf='RandomResizedCrop(448, scale=(0.5,1)) + RandomHorizontalFlip',
              eval_tf='Resize((448,448)) bilinear', init='torchvision IMAGENET1K_V1')


class LNPost(nn.Module):
    """verbatim theory_sweeps_lib.LNPost: fc(LN(mean_t a_t))."""
    def __init__(s, k=10, c=2048):
        super().__init__(); s.ln = nn.LayerNorm(c); s.fc = nn.Linear(c, k)

    def forward(s, A): return s.fc(s.ln(A.mean(-1)))


def build_head(h):
    if h == 'H3':
        return LNPost(K, C)
    return HD.build(HEAD_NAME[h], k=K, c=C)


class Net(nn.Module):
    """torchvision ResNet-50 up to layer4 (14x14 grid at 448) + a head on (B,C,T)."""
    def __init__(s, h, pretrained=True):
        super().__init__()
        import torchvision
        wts = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet50(weights=wts)
        s.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                   r.layer1, r.layer2, r.layer3, r.layer4)
        s.head = build_head(h)

    def cells(s, x):
        f = s.backbone(x)                     # (B,C,H,W)
        return f.flatten(2)                   # (B,C,T)

    def forward(s, x):
        A = s.cells(x)
        with torch.autocast('cuda', enabled=False):
            return s.head(A.float())


def tfs():
    from torchvision import transforms as TT
    ip = TT.InterpolationMode.BILINEAR
    tr = TT.Compose([TT.RandomResizedCrop(SIZE, scale=(0.5, 1.0), interpolation=ip),
                     TT.RandomHorizontalFlip(), TT.ToTensor(), TT.Normalize(MEAN, STD)])
    ev = TT.Compose([TT.Resize((SIZE, SIZE), interpolation=ip), TT.ToTensor(),
                     TT.Normalize(MEAN, STD)])
    return tr, ev


class DS(torch.utils.data.Dataset):
    def __init__(s, recs, tf):
        s.p = [r['path'] for r in recs]; s.y = [r['label'] for r in recs]; s.tf = tf

    def __len__(s): return len(s.p)

    def __getitem__(s, i):
        from PIL import Image
        return s.tf(Image.open(s.p[i]).convert('RGB')), s.y[i], i


def ckpt_path(h, seed):
    return os.path.join(CKPT, 'K2_%s_s%d.pt' % (h, seed))


def tag(h, seed):
    return '%s_s%d' % (h, seed)


def bank_indices(n_fit):
    """the 8 training-split references of the earlier localisation arms (w2_run.py)."""
    rs = np.random.RandomState(W.PROTOCOL['seed'])
    return sorted(rs.choice(n_fit, W.PROTOCOL['n_bank'], replace=False).tolist())


def save_json(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=W._jd)
    return path
