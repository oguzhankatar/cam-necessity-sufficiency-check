"""torchvision 0.21 classifiers exposed with timm's forward_features / forward_head split.

The split is placed at the last feature map of the trunk; forward_head is the rest of
torchvision's own forward, transcribed from the torchvision 0.21 source of each family
(no operation added, removed or reordered; DenseNet's in-place ReLU acts on a clone so
the certificate's cell matrix is not overwritten). Random weights (weights=None).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


def family(name):
    for k in ('alexnet', 'vgg', 'resnet', 'resnext', 'wide_resnet', 'regnet', 'densenet', 'mobilenet_v2',
              'mobilenet_v3', 'efficientnet', 'mnasnet', 'shufflenet', 'squeezenet', 'googlenet',
              'inception_v3', 'convnext', 'swin', 'maxvit', 'vit'):
        if name.startswith(k):
            return {'resnext': 'resnet', 'wide_resnet': 'resnet'}.get(k, k)
    raise KeyError(name)


class TV(nn.Module):
    def __init__(s, name):
        super().__init__()
        s.tvname, s.fam = name, family(name)
        kw = {}
        if s.fam in ('googlenet', 'inception_v3'):
            kw = dict(aux_logits=False, init_weights=True)   # aux heads are unused in eval
        s.m = M.get_model(name, weights=None, **kw)
        size = 299 if s.fam == 'inception_v3' else (s.m.image_size if s.fam == 'vit' else 224)
        s.pretrained_cfg = dict(input_size=(3, size, size))
        s.num_prefix_tokens = 1 if s.fam == 'vit' else 0
        with torch.no_grad():
            f = s.forward_features(torch.zeros(1, 3, size, size))
        s.num_features = f.shape[-1] if s.fam in ('vit', 'swin') else f.shape[1]

    def forward_features(s, x):
        m, fam = s.m, s.fam
        if fam in ('alexnet', 'vgg', 'densenet', 'mobilenet_v2', 'mobilenet_v3', 'efficientnet',
                   'squeezenet', 'convnext', 'swin'):
            return m.features(x)
        if fam == 'resnet':
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))
        if fam == 'regnet':
            return m.trunk_output(m.stem(x))
        if fam == 'mnasnet':
            return m.layers(x)
        if fam == 'shufflenet':
            x = m.maxpool(m.conv1(x))
            return m.conv5(m.stage4(m.stage3(m.stage2(x))))
        if fam == 'googlenet':
            x = m._transform_input(x)
            x = m.maxpool1(m.conv1(x)); x = m.maxpool2(m.conv3(m.conv2(x)))
            x = m.maxpool3(m.inception3b(m.inception3a(x)))
            x = m.inception4e(m.inception4d(m.inception4c(m.inception4b(m.inception4a(x)))))
            return m.inception5b(m.inception5a(m.maxpool4(x)))
        if fam == 'inception_v3':
            x = m._transform_input(x)
            x = m.maxpool1(m.Conv2d_2b_3x3(m.Conv2d_2a_3x3(m.Conv2d_1a_3x3(x))))
            x = m.maxpool2(m.Conv2d_4a_3x3(m.Conv2d_3b_1x1(x)))
            x = m.Mixed_6a(m.Mixed_5d(m.Mixed_5c(m.Mixed_5b(x))))
            x = m.Mixed_6e(m.Mixed_6d(m.Mixed_6c(m.Mixed_6b(x))))
            return m.Mixed_7c(m.Mixed_7b(m.Mixed_7a(x)))
        if fam == 'maxvit':
            x = m.stem(x)
            for b in m.blocks:
                x = b(x)
            return x
        if fam == 'vit':
            x = m._process_input(x)
            n = x.shape[0]
            x = torch.cat([m.class_token.expand(n, -1, -1), x], dim=1)
            return m.encoder(x)
        raise KeyError(fam)

    def forward_head(s, x):
        m, fam = s.m, s.fam
        if fam in ('alexnet', 'vgg'):
            return m.classifier(torch.flatten(m.avgpool(x), 1))
        if fam in ('resnet', 'regnet'):
            return m.fc(torch.flatten(m.avgpool(x), 1))
        if fam == 'densenet':
            out = F.relu(x.clone(), inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            return m.classifier(torch.flatten(out, 1))
        if fam == 'mobilenet_v2':
            x = F.adaptive_avg_pool2d(x, (1, 1))
            return m.classifier(torch.flatten(x, 1))
        if fam in ('mobilenet_v3', 'efficientnet', 'convnext'):
            return m.classifier(torch.flatten(m.avgpool(x), 1)) if fam != 'convnext' else m.classifier(m.avgpool(x))
        if fam in ('mnasnet', 'shufflenet'):
            x = x.mean([2, 3])
            return m.classifier(x) if fam == 'mnasnet' else m.fc(x)
        if fam == 'squeezenet':
            return torch.flatten(m.classifier(x), 1)
        if fam == 'googlenet':
            x = torch.flatten(m.avgpool(x), 1)
            return m.fc(m.dropout(x))
        if fam == 'inception_v3':
            x = m.dropout(m.avgpool(x))
            return m.fc(torch.flatten(x, 1))
        if fam == 'swin':
            x = m.norm(x); x = m.permute(x); x = m.avgpool(x); x = m.flatten(x)
            return m.head(x)
        if fam == 'maxvit':
            return m.classifier(x)
        if fam == 'vit':
            return m.heads(x[:, 0])
        raise KeyError(fam)

    def forward(s, x):
        return s.forward_head(s.forward_features(x))


def tv_names():
    return M.list_models(module=M)
