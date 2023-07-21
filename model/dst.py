
import torch.nn as nn

from module.utils import create_PositionalEncoding, add_position, _get_activation_fn
from module.dst_layer import DST_Encoder

class DST(nn.Module):
    def __init__(self, 
                input_dim       = 1024,
                length          = 326,
                ffn_embed_dim   = 512, 
                num_layers      = 4, 
                num_heads       = 8, 
                num_classes     = 4, 
                dropout         = 0.1,
                bias            = True,
                activation      = 'relu'):
        super().__init__()
        
        self.position = create_PositionalEncoding(input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        hop = 20
        duration = [50, 1000]
        duration_init_min = duration[0] / hop / 2  # half of duration: / 2
        duration_init_max = duration[1] / hop / 2
        
        self.layers = nn.ModuleList(
            [DST_Encoder(input_dim, ffn_embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout, bias, activation) 
            for _ in range(num_layers)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            _get_activation_fn(activation, module=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, input_dim//4),
            _get_activation_fn(activation, module=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//4, num_classes),
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for m in self.modules():
            if hasattr(m, '_reset_specially'):
                m._reset_specially()

    def forward(self, x):
        x = self.input_norm(x)
        x = add_position(x, self.position)

        for layer in self.layers:
            x = layer(x)
        
        pred = self.classifier(x)
        pred = self.avgpool(pred.transpose(-1, -2)).squeeze(dim=-1)

        return pred

