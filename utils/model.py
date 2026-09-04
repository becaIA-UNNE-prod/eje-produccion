import torch
import torch.nn as nn

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

class SimpleUNet(nn.Module):
    """U-Net de 4 niveles de encoder/decoder (64-128-256-512, cuello de
    botella 1024). Se profundizó desde una versión de 2 niveles porque el
    problema real (28 canales = 7 meses x 4 bandas de Sentinel-2, clases
    minoritarias como mani/sorgo) necesita más campo receptivo y capacidad
    que una U-Net chica -- con patches de 256x256, 2 niveles solo baja hasta
    64x64 en el cuello de botella; con 4 niveles baja hasta 16x16.
    Dropout2d en los niveles más profundos (donde hay más parámetros y más
    riesgo de sobreajuste) como regularización, dado que antes no había
    ninguna más allá de weight_decay en el optimizer.
    """
    def __init__(self, in_channels, num_classes):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Cuello de botella
        self.bottleneck = DoubleConv(512, 1024, dropout=0.3)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512, dropout=0.2)  # 512 + 512 (skip)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)  # 256 + 256 (skip)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 + 128 (skip)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # 64 + 64 (skip)

        # Salida
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        b = self.bottleneck(p4)

        u1 = self.up1(b)
        u1 = torch.cat([u1, c4], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, c3], dim=1)
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, c2], dim=1)
        d3 = self.dec3(u3)

        u4 = self.up4(d3)
        u4 = torch.cat([u4, c1], dim=1)
        d4 = self.dec4(u4)

        out = self.out_conv(d4)
        return out


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


class TransUNet(nn.Module):
    """SimpleUNet con el cuello de botella aumentado por self-attention global.

    Mismo encoder/decoder/skip connections que SimpleUNet (misma cantidad de
    niveles y canales) para que la comparacion contra el baseline aisle el
    efecto de agregar atencion global, sin mezclar otros cambios de
    arquitectura. Unica diferencia: despues del DoubleConv del bottleneck
    (512->1024, 16x16 con patches de 256x256), un TransformerBottleneck le da
    a cada posicion espacial contexto de todo el patch en una sola pasada,
    algo que la CNN solo logra indirectamente apilando muchas capas.
    """
    def __init__(self, in_channels, num_classes,
                 transformer_depth=2, transformer_heads=8, transformer_dropout=0.1):
        super(TransUNet, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Cuello de botella: DoubleConv + atencion global
        self.bottleneck = DoubleConv(512, 1024, dropout=0.3)
        self.transformer = TransformerBottleneck(
            1024, depth=transformer_depth, num_heads=transformer_heads,
            dropout=transformer_dropout, base_grid=16)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512, dropout=0.2)  # 512 + 512 (skip)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)  # 256 + 256 (skip)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 + 128 (skip)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # 64 + 64 (skip)

        # Salida
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        b = self.bottleneck(p4)
        b = self.transformer(b)

        u1 = self.up1(b)
        u1 = torch.cat([u1, c4], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, c3], dim=1)
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, c2], dim=1)
        d3 = self.dec3(u3)

        u4 = self.up4(d3)
        u4 = torch.cat([u4, c1], dim=1)
        d4 = self.dec4(u4)

        out = self.out_conv(d4)
        return out
