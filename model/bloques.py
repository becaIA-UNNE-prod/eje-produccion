import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(DoubleConv, self).__init__()
        capas = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            capas.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*capas)

    def forward(self, x):
        return self.conv(x)


class TransformerBottleneck(nn.Module):
    """Self-attention global sobre el mapa de features del cuello de botella.

    Aplana el mapa (B, dim, H, W) a tokens (B, H*W, dim) y los pasa por un
    TransformerEncoder estandar (pre-norm, GELU). El positional embedding se
    aprende para una grilla base (16x16, la que resulta de un patch de
    entrenamiento 256x256 tras 4 poolings) y se interpola por bicubic si en
    inferencia la grilla es otra (p.ej. 512x512 -> 32x32), igual que hacen
    ViT/DeiT al cambiar de resolucion.
    """
    def __init__(self, dim, depth=2, num_heads=8, mlp_ratio=2.0, dropout=0.1, base_grid=16):
        super().__init__()
        self.base_grid = base_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, base_grid, base_grid))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        capa = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(capa, num_layers=depth)

    def _pos_embed_para(self, H, W):
        if (H, W) == (self.base_grid, self.base_grid):
            return self.pos_embed
        return nn.functional.interpolate(
            self.pos_embed, size=(H, W), mode="bicubic", align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pos = self._pos_embed_para(H, W)
        tokens = (x + pos).flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)
