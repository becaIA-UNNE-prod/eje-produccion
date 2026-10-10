import torch
import torch.nn as nn
import torch.nn.functional as F
from .bloques import DoubleConv


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder (Garnot & Landrieu 2020),
    aplicado en paralelo sobre cada posicion espacial de un mapa de features
    (B, T, C, H, W) -- un "token" por mes, no por parche espacial.

    A diferencia de self-attention estandar (una query por paso temporal,
    costo O(T^2)), usa una unica "master query" aprendida por cabeza que NO
    depende del input -- pensada para T chico (7 meses) y para que el costo
    de atencion sea O(T). El positional embedding es aprendido por indice de
    mes (0..T-1) en vez de sinusoidal por dia-del-anio: los composites
    mensuales de 1_train_cordoba.py siempre vienen en el mismo calendario
    fijo (MESES_FIJOS), no hay fechas irregulares que codificar.

    Devuelve, ademas de la salida colapsada, los mapas de atencion por
    cabeza (B, n_heads, T, H, W) SIN promediar -- U-TAE (Garnot & Landrieu
    2021) reusa esos mismos pesos (reescalados por interpolacion bilineal)
    para colapsar el tiempo en los skip connections de resoluciones mas
    finas del encoder, evitando correr atencion de nuevo en cada escala.
    """
    def __init__(self, in_channels, T, n_heads=8, d_k=None, dropout=0.1):
        super().__init__()
        self.T = T
        self.n_heads = n_heads
        self.d_k = d_k or in_channels // n_heads
        d_model = self.n_heads * self.d_k

        self.k_proj = nn.Conv2d(in_channels, d_model, 1)
        self.v_proj = nn.Conv2d(in_channels, d_model, 1)
        self.pos_embed = nn.Parameter(torch.zeros(T, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.master_query = nn.Parameter(torch.zeros(n_heads, self.d_k))
        nn.init.trunc_normal_(self.master_query, std=0.02)

        self.out_proj = nn.Conv2d(d_model, in_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        xt = x.reshape(B * T, C, H, W)
        k = self.k_proj(xt).view(B, T, self.n_heads, self.d_k, H, W)
        v = self.v_proj(xt).view(B, T, self.n_heads, self.d_k, H, W)
        k = k + self.pos_embed.view(1, T, self.n_heads, self.d_k, 1, 1)

        q = self.master_query.view(1, 1, self.n_heads, self.d_k, 1, 1)
        scores = (q * k).sum(dim=3, keepdim=True) / (self.d_k ** 0.5)  # (B,T,heads,1,H,W)
        attn = torch.softmax(scores, dim=1)  # softmax sobre el eje temporal

        out = (attn * v).sum(dim=1)  # (B, heads, d_k, H, W)
        out = out.reshape(B, self.n_heads * self.d_k, H, W)
        out = self.out_proj(self.dropout(out))
        out = self.norm(out)

        attn_maps = attn.squeeze(3).permute(0, 2, 1, 3, 4)  # (B, heads, T, H, W)
        return out, attn_maps


def _upsample_attn(attn, size):
    """attn: (B, n_heads, T, He, We) -> mismo tensor interpolado a `size`.

    Como la interpolacion es lineal y se aplica con los mismos coeficientes
    a cada slice temporal, la suma sobre T se preserva en 1 en cada posicion
    de salida (ya vale 1 en la entrada, por el softmax de LTAE2d) sin
    necesidad de renormalizar.
    """
    B, n_heads, T, He, We = attn.shape
    attn = attn.reshape(B * n_heads * T, 1, He, We)
    attn = F.interpolate(attn, size=size, mode="bilinear", align_corners=False)
    return attn.reshape(B, n_heads, T, *size)


def _reponderar_temporal(feat, attn):
    """Colapsa el eje temporal de `feat` (B, T, C, H, W) repartiendo sus C
    canales en n_heads grupos, cada uno ponderado por los pesos de atencion
    de su cabeza (`attn`, ya normalizados sobre T y en la misma resolucion
    H, W que `feat`)."""
    B, T, C, H, W = feat.shape
    n_heads = attn.shape[1]
    feat = feat.view(B, T, n_heads, C // n_heads, H, W)
    pesos = attn.permute(0, 2, 1, 3, 4).unsqueeze(3)  # (B, T, n_heads, 1, H, W)
    return (feat * pesos).sum(dim=1).reshape(B, C, H, W)


class UTAE(nn.Module):
    """U-TAE (Sainte Fare Garnot & Landrieu 2021), sobre el mismo esqueleto
    de encoder/decoder que SimpleUNet (64-128-256-512, cuello de botella
    1024) para que la comparacion aisle el efecto arquitectonico real.

    Diferencia central con SimpleUNet/TransUNet: en vez de aplanar los 7
    meses x 4 bandas en 28 canales sueltos desde el primer Conv2d (con lo
    que la red ve "mas bandas espectrales", perdiendo el orden temporal), la
    entrada se trata como una serie de T=7 imagenes de 4 bandas. El mismo
    encoder (pesos compartidos) se aplica a cada mes por separado (T se
    pliega en la dimension de batch); LTAE2d colapsa el tiempo en el cuello
    de botella con self-attention, y esos mismos mapas de atencion (subidos
    de resolucion) se reusan para colapsar el tiempo en los 4 niveles de
    skip connections, en vez de correr atencion de nuevo en cada escala. El
    decoder es identico al de SimpleUNet.

    Encaja con el problema real: la fenologia (como cambia mes a mes la
    firma espectral de un lote) es la señal que distingue Maiz/Soja/Mani, y
    tratarla explicitamente como secuencia (en vez de como canales sueltos
    sin orden) es lo que un modelo channel-stacked no puede aprovechar
    aunque tenga mas capacidad (ver TransUNet: la atencion espacial en el
    bottleneck no alcanza a compensar esta perdida de estructura).

    Costo: el encoder corre T veces (una por mes) en vez de una sola vez
    sobre 28 canales combinados -- mas lento que SimpleUNet/TransUNet en
    FLOPs de encoder, pero con menos parametros nuevos (LTAE2d es liviana).
    """
    def __init__(self, in_channels, num_classes, T=7, n_heads=8, dropout=0.1):
        super(UTAE, self).__init__()
        if in_channels % T != 0:
            raise ValueError(f"in_channels={in_channels} no es multiplo de T={T}")
        self.T = T

        c_por_mes = in_channels // T
        # Encoder (pesos compartidos entre meses -- se aplica con T plegado en el batch)
        self.conv1 = DoubleConv(c_por_mes, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024, dropout=0.3)
        self.ltae = LTAE2d(1024, T=T, n_heads=n_heads, dropout=dropout)

        # Decoder (identico a SimpleUNet: opera sobre features ya colapsadas en tiempo)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512, dropout=0.2)  # 512 + 512 (skip)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)  # 256 + 256 (skip)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 + 128 (skip)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)  # 64 + 64 (skip)

        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        B, C_flat, H, W = x.shape
        T = self.T
        c_por_mes = C_flat // T
        # (B, 28, H, W) -> (B, T, 4, H, W) -- ver 03c_generar_dataset_cordoba_f16.py:
        # los canales estan concatenados mes a mes (mismo orden que MESES_FIJOS),
        # 4 bandas por mes, asi que este reshape recupera exactamente esa estructura.
        xt = x.view(B, T, c_por_mes, H, W).reshape(B * T, c_por_mes, H, W)

        c1 = self.conv1(xt); p1 = self.pool1(c1)
        c2 = self.conv2(p1); p2 = self.pool2(c2)
        c3 = self.conv3(p2); p3 = self.pool3(c3)
        c4 = self.conv4(p3); p4 = self.pool4(c4)
        bott = self.bottleneck(p4)

        def unfold(t):
            _, c, h, w = t.shape
            return t.view(B, T, c, h, w)

        c1, c2, c3, c4, bott = unfold(c1), unfold(c2), unfold(c3), unfold(c4), unfold(bott)

        b_col, attn = self.ltae(bott)  # (B, 1024, He, We), (B, n_heads, T, He, We)

        def colapsar(feat):
            attn_up = _upsample_attn(attn, feat.shape[-2:])
            return _reponderar_temporal(feat, attn_up)

        c1c, c2c, c3c, c4c = colapsar(c1), colapsar(c2), colapsar(c3), colapsar(c4)

        u1 = self.up1(b_col)
        u1 = torch.cat([u1, c4c], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, c3c], dim=1)
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, c2c], dim=1)
        d3 = self.dec3(u3)

        u4 = self.up4(d3)
        u4 = torch.cat([u4, c1c], dim=1)
        d4 = self.dec4(u4)

        return self.out_conv(d4)
