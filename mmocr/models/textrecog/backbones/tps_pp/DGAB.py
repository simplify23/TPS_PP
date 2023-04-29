import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DGAB_Block(nn.Module):
    def __init__(self, dim, point=8, qkv_bias=False, height=1,width=63, proj_drop=0.):
        super().__init__()

        self.mlp_h = nn.Sequential(
                    nn.Linear(height + point, height + 1, bias=qkv_bias),
                                   )
        self.mlp_w = nn.Sequential(
                    nn.Linear(width + point, width + 1, bias=qkv_bias),
                                   )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        y = rearrange(y, 'b t c -> b c t')
        # b c w  , b c h
        w = self.mlp_w(torch.cat([x.mean(2), y], 2))
        v_w = w[:,:,:-1].softmax(dim=-1).unsqueeze(2)

        h = self.mlp_h(torch.cat([x.mean(3), y], 2))
        v_h = h[:,:,:-1].softmax(dim=-1).unsqueeze(3)

        # x = h * x + w * x
        # x = v_w * x * w[:, :, -1].unsqueeze(-1).unsqueeze(-1)
        x = v_h * x * h[:,:,-1].unsqueeze(-1).unsqueeze(-1) + v_w * x * w[:,:,-1].unsqueeze(-1).unsqueeze(-1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DGAB(nn.Module):

    def __init__(self, dim, mlp_ratio=4., width = 128, high = 32, point=16, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=DGAB_Block):
        super().__init__()
        tuple_dim = [high,width]
        # tuple_dim = dim
        self.norm1 = norm_layer(tuple_dim)
        self.attn = mlp_fn(dim, point=point, width=width,height=high,qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(tuple_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), y)) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x