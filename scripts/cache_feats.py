"""Cache frozen ResNet50 layer4 feature maps for imagenette (train+val).

Everything downstream (aggregator ablation, Sobol on the head, training
dynamics) re-uses these so the BACKBONE is held fixed and the only thing
that varies is the AGGREGATOR. Output: (N,2048,7,7) fp16 + int labels.
"""
import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

sys.path.insert(0, os.path.dirname(__file__))
from pilot import get_data

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'outputs')
dev = 'cuda'


def main():
    torch.set_grad_enabled(False)
    root = get_data(os.path.join(ROOT, 'data'))
    tf = transforms.Compose([
        transforms.Resize(160), transforms.CenterCrop(160), transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to(dev)
    body = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                               m.layer1, m.layer2, m.layer3, m.layer4)
    for split in ('train', 'val'):
        fp = os.path.join(OUT, f'feat_{split}.npy')
        if os.path.exists(fp):
            print('exists', fp); continue
        ds = datasets.ImageFolder(os.path.join(root, split), tf)
        ld = DataLoader(ds, batch_size=24, num_workers=0)
        F, Y = [], []
        for i, (x, y) in enumerate(ld):
            F.append(body(x.to(dev)).half().cpu().numpy()); Y.append(y.numpy())
            if i % 50 == 0:
                print(f'  {split} {i*24}/{len(ds)}', flush=True)
        F = np.concatenate(F); Y = np.concatenate(Y)
        np.save(fp, F); np.save(os.path.join(OUT, f'lab_{split}.npy'), Y)
        print(split, F.shape, F.dtype, 'classes', len(set(Y.tolist())))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
