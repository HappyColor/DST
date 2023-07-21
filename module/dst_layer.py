
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from module.utils import _get_activation_fn, arbitrary_segment_mask, arbitrary_point_mask, inverse_sigmoid

class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, dropout=0., activation='relu'):
        super().__init__()
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        return x

class Deformable_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.length = length
        self.init_duration_min = duration_init_min
        self.init_duration_max = duration_init_max

        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5

        self.offset_duration = nn.Linear(embed_dim, num_heads * 2, bias=True)
        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias)
    
    def _reset_specially(self):
        nn.init.constant_(self.offset_duration.weight.data, 0.)

        offset_init = torch.zeros(self.num_heads).type(torch.float)
        duration_init = torch.arange(self.init_duration_min, self.init_duration_max+1, \
            step=(self.init_duration_max-self.init_duration_min)/(self.num_heads-1)) / self.length

        duration_init = torch.tensor(list(map(inverse_sigmoid, duration_init)))

        bias_init = torch.cat((offset_init, duration_init), dim=-1)
        with torch.no_grad():
            self.offset_duration.bias = nn.Parameter(bias_init.view(-1))

    def query_index(self, bsz, dtype, device):
        return torch.arange(0, self.length, dtype=dtype, device=device)[None, :, None].expand(bsz, -1, -1)

    def forward(self, x: torch.Tensor):
        # b t c
        Q, K, V = self.qkv(x).chunk(3, dim=-1)

        # b t nh
        offset_duration = self.offset_duration(Q) #* self.length
        offset, duration = offset_duration.chunk(2, dim=-1)
        offset = offset.tanh_() * self.length
        duration = duration.sigmoid_() * self.length

        query_index = self.query_index(x.shape[0], x.dtype, x.device)
        anchor = query_index + offset

        # (b nh) t
        anchor = einops.rearrange(anchor, 'b t nh -> (b nh) t', nh=self.num_heads)
        duration = einops.rearrange(duration, 'b t nh -> (b nh) t', nh=self.num_heads)#.abs_()

        # (b nh) t
        start = anchor - duration
        end = anchor + duration
        bound_l_idx_l = torch.floor(start)
        bound_r_idx_r = torch.ceil(end)
        anchor_idx_l = torch.floor(anchor)
        anchor_idx_r = anchor_idx_l + 1

        # (b nh) t
        neg_bound_l_dist_l = bound_l_idx_l - start
        neg_bound_r_dist_r = end - bound_r_idx_r
        anchor_dist_l = anchor - anchor_idx_l
        anchor_dist_r = 1 - anchor_dist_l
        
        # (b nh) t t   requires_grad is False
        bound_l_mask_l = arbitrary_point_mask(index=bound_l_idx_l, len_out=self.length, return_bool=False)
        bound_r_mask_r = arbitrary_point_mask(index=bound_r_idx_r, len_out=self.length, return_bool=False)
        anchor_l_mask = arbitrary_point_mask(index=anchor_idx_l, len_out=self.length, return_bool=False)
        anchor_r_mask = arbitrary_point_mask(index=anchor_idx_r, len_out=self.length, return_bool=False)
        
        # (b nh) t t
        neg_bound_weight = neg_bound_l_dist_l[:, :, None] * bound_l_mask_l + neg_bound_r_dist_r[:, :, None] * bound_r_mask_r
        anchor_weight = anchor_dist_l[:, :, None] * anchor_r_mask + anchor_dist_r[:, :, None] * anchor_l_mask
        point_weight = 1 + neg_bound_weight + anchor_weight

        # (b nh) t t
        attn_mask = arbitrary_segment_mask(start_index=bound_l_idx_l, end_index=bound_r_idx_r, len_out=self.length)

        # (b nh) t hd
        Q = einops.rearrange(Q, 'b t (nh hd) -> (b nh) t hd', nh=self.num_heads)
        K = einops.rearrange(K, 'b t (nh hd) -> (b nh) t hd', nh=self.num_heads)
        V = einops.rearrange(V, 'b t (nh hd) -> (b nh) t hd', nh=self.num_heads)

        # (b nh) t t
        attn_weights = torch.einsum('...ic, ...rc -> ...ir', Q, K) * self.scaling
        attn_weights = attn_weights * point_weight
        attn_weights = attn_weights.masked_fill(attn_mask, -1e8)  # float('-inf')
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('...ir, ...rc -> ...ic', attn_weights, V)
        output = einops.rearrange(output, '(b nh) t hd -> b t (nh hd)', nh=self.num_heads)

        # b t c
        output = self.project_out(output)

        return output

class DST_Encoder(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout=0., bias=True, activation='relu'):
        super().__init__()
        self.dropout = dropout

        self.attention = Deformable_Attention(embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout, bias)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_embed_dim, dropout, activation)

    def perform_attn(self, x):
        residual = x
        x = self.attention(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_norm(x)
        return x

    def perform_ffn(self, x):
        return self.ffn(x)

    def forward(self, x):
        x = self.perform_attn(x)
        x = self.perform_ffn(x)
        return x
