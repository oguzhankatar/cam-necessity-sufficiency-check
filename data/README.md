# Data

All datasets are public and none is redistributed here. The scripts fetch them from their public sources or expect them in the locations below. Only `data/nih/` is part of the repository: the train and test split lists of NIH ChestX-ray14 used in the study and the list of images that were fetched.

| Dataset | Used for | Fetched or read by | Location |
|---|---|---|---|
| ImageNet-1k validation (a public re-encoded mirror) | top-1 accuracy of the normalisation edits, operator sweeps on 224 images | `scripts/get_in1k.py` | `IN1K_PARQUET`, default `<temp>/nscam_cache/in1k_parq` |
| Imagenette, Imagewoof, CIFAR-100 | operator sweeps, end-to-end sweep | `scripts/pilot.py`, `scripts/nslib.py` | `R2CACHE`, default `<temp>/nscam_cache/r2cache` |
| CUB-200-2011, PASCAL VOC 2012, MS-COCO val2017 | localisation | `scripts/wsol/w0_data.py`, `scripts/wsol/wlib.py` | `data/` of this repository |
| NIH ChestX-ray14 (with the box annotations), RSNA Pneumonia Detection Challenge | medical arms | `scripts/medical/get_nih.py`, `get_nih2.py`, `get_rsna.py` | `MEDCACHE`, default `<temp>/nscam_cache/medcache` |
| ISIC 2017 and 2018 | medical arm with segmentation masks | `scripts/medical/m5_isic.py`, `analyses/isic_three_seeds/scripts/k3_seed.py` | `MEDCACHE` |

`<temp>` is the system temporary directory. Pretrained weights are downloaded by timm, torchvision and TorchXRayVision on first use.

Please cite the datasets when you use them: Wah et al. (2011) for CUB-200-2011, Everingham et al. (2010) for PASCAL VOC, Lin et al. (2014) for MS-COCO, Russakovsky et al. (2015) for ImageNet, Wang et al. (2017) for NIH ChestX-ray14, Shih et al. (2019) for RSNA Pneumonia, and Codella et al. (2018, 2019) for ISIC.
