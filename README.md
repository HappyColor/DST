# DST
![DST_framework](./figures/framework.png)
[\[ICASSP 2023 Paper\]](https://ieeexplore.ieee.org/abstract/document/10096966) DST: Deformable Speech Transformer for Emotion Recognition

## Usage
```
import torch
from model.dst import DST

kwargs = {
    "input_dim": 1024,
    "length": 326,
    "ffn_embed_dim": 512,
    "num_layers": 4,
    "num_heads": 8,
    "num_classes": 4,
    "dropout": 0.1,
    "bias": True,
    "activation": "relu"
    }

model = DST(**kwargs)

input = torch.randn(1, kwargs['length'], kwargs['input_dim'])
output = model(input)  # output shape: (1, kwargs['num_classes'])
```

## Citation
Please cite the following paper if you feel DST useful to your research
```
@INPROCEEDINGS{chen2023DST,
  author={Chen, Weidong and Xing, Xiaofen and Xu, Xiangmin and Pang, Jianxin and Du, Lan},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={DST: Deformable Speech Transformer for Emotion Recognition}, 
  year={2023}
  }
```