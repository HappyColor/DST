# DST
![DST_framework](./figures/framework.png)
[\[ICASSP 2023 Paper\]](https://ieeexplore.ieee.org/abstract/document/10096966) DST: Deformable Speech Transformer for Emotion Recognition

# Usage
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