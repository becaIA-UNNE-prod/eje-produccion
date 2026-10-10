import torch
import torch.nn as nn
import torch.nn.functional as F
from .bloques import DoubleConv, TransformerBottleneck


class SelfAttentionConv(nn.Module):
    """Self-Attention no-local (la "SA" de FEM en FCR-TransUNet, Han et al.
    2026): cada posicion espacial del mapa de features atiende a todo el
    resto del mapa. Proyecciones q/k/v con conv dilatada (dilation=2, "las
    convoluciones dilatadas expanden el campo receptivo" segun el paper) para
    que cada token ya integre algo de contexto local antes de la atencion
    global.

    A diferencia de un transformer estandar no hay FFN despues (el paper solo
    describe SA(B,T) = B*T): la salida es directamente la mezcla ponderada de
    V, con un residual para no descartar la feature original si la atencion
    no aporta nada util en ese nivel.

    K/V se resumen con un adaptive-pool a una grilla chica (kv_pool) antes de
    la atencion -- a diferencia del bottleneck (TransformerBottleneck, que
    corre sobre 16x16=256 tokens), FEM se aplica en el nivel 3 del encoder
    (64x64=4096 tokens con patches de 256x256): atencion completa ahi
    (4096x4096) es demasiado pesada en memoria a batch grande. Pooling
    "asimetrico" de K/V (Q se queda a resolucion completa) es una
    simplificacion practica no descripta explicitamente en el resumen del
    paper que tuve disponible, pero preserva la idea (cada posicion atiende
    a un resumen global) a un costo manejable.
    """
    def __init__(self, channels, n_heads=8, dilation=2, kv_pool=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = channels // n_heads
        self.q_proj = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.k_proj = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.v_proj = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.pool_kv = nn.AdaptiveAvgPool2d(kv_pool)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_proj(x)
        k = self.pool_kv(self.k_proj(x))
        v = self.pool_kv(self.v_proj(x))
        Hk, Wk = k.shape[-2:]

        q = q.view(B, self.n_heads, self.d_k, H * W)
        k = k.view(B, self.n_heads, self.d_k, Hk * Wk)
        v = v.view(B, self.n_heads, self.d_k, Hk * Wk)

        attn = torch.softmax(
            torch.einsum("bhdi,bhdj->bhij", q, k) / (self.d_k ** 0.5), dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(B, C, H, W)
        out = self.out_proj(out)
        return self.norm(x + out)


class GatedConv(nn.Module):
    """Gated Convolution (la "GC" de FEM): un gate derivado de la feature
    "alta" (en FCR-TransUNet, la salida de SA -- contexto global) modula la
    feature "baja" (la entrada original a FEM, con detalle local) -- deja
    pasar la señal local donde el contexto global la respalda, y la atenua
    donde parece ruido de fondo.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.feat_conv = nn.Conv2d(channels, channels, kernel_size, padding=pad)

    def forward(self, f_low, f_high):
        gate = torch.sigmoid(self.gate_conv(f_high))
        feat = F.relu(self.feat_conv(f_low))
        return feat * gate


class FeatureEnhancementModule(nn.Module):
    """FEM de FCR-TransUNet: SA (self-attention no-local) + GC (gated conv)
    en un unico nivel intermedio del encoder -- suprime ruido de fondo y
    refuerza contexto antes de bajar a niveles mas profundos, donde ya se
    pierde demasiado detalle espacial para los bordes finos entre lotes.
    """
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.sa = SelfAttentionConv(channels, n_heads=n_heads)
        self.gc = GatedConv(channels)

    def forward(self, x):
        f_high = self.sa(x)
        return self.gc(f_low=x, f_high=f_high)


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou, Zhou & Feng, CVPR 2021), como la usa
    FCR-TransUNet en los 4 skip connections: a diferencia de CBAM (que agrega
    todo el mapa espacial a un solo vector por canal, perdiendo posicion),
    hace pooling promedio por separado a lo largo de H y de W -- conserva "en
    que fila"/"en que columna" esta la activacion. Pensado para distinguir
    lotes espectralmente parecidos (Maiz/Soja) por su forma o disposicion
    espacial, no solo por su intensidad promedio de canal (que es lo unico
    que ve un pooling global).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        zh = x.mean(dim=3, keepdim=True)                       # (B,C,H,1) -- promedio sobre W
        zw = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # (B,C,W,1) -- promedio sobre H

        z = torch.cat([zh, zw], dim=2)          # (B,C,H+W,1)
        z = self.act(self.bn(self.conv1(z)))
        zh, zw = torch.split(z, [H, W], dim=2)
        zw = zw.permute(0, 1, 3, 2)              # (B,mid,1,W)

        gh = torch.sigmoid(self.conv_h(zh))      # (B,C,H,1)
        gw = torch.sigmoid(self.conv_w(zw))      # (B,C,1,W)
        return x * gh * gw


class ReverseAttention(nn.Module):
    """Reverse Attention (PraNet, Fan et al. 2020), como la usa FCR-TransUNet
    en la segunda etapa del decoder: en vez de fusionar el skip connection de
    forma directa, el propio decoder predice un mapa de "borde" (que tan
    seguro esta de esa zona), lo invierte, y usa esa inversa para pesar el
    skip -- asi resalta las zonas donde el decoder todavia esta indeciso
    (tipicamente el limite entre lotes, donde mas se equivoca un clasificador
    de cultivos) en vez de las que ya viene clasificando con confianza.
    """
    def __init__(self, dec_channels):
        super().__init__()
        self.edge_conv = nn.Conv2d(dec_channels, 1, 1)

    def forward(self, feat_dec, feat_enc):
        p_edge = torch.sigmoid(self.edge_conv(feat_dec))
        r = (1 - p_edge) * feat_enc
        return torch.cat([feat_dec, r], dim=1)


class FCRTransUNet(nn.Module):
    """FCR-TransUNet (Han et al. 2026, MDPI Agriculture 16(7):727), adaptado
    a nuestro esqueleto de 4 niveles (64-128-256-512, bottleneck 1024) para
    que la comparacion contra SimpleUNet/TransUNet/UTAE aisle el efecto
    arquitectonico. Reconstruido a partir de un resumen extraido del paper
    (sin acceso directo al PDF/ecuaciones completas) -- es una adaptacion
    fiel a la descripcion textual de cada modulo, no una replica linea por
    linea; ver docstrings de SelfAttentionConv/CoordinateAttention/
    ReverseAttention para las simplificaciones puntuales que fueron
    necesarias.

    Tres piezas nuevas respecto a TransUNet (que ya aporta el bottleneck
    transformer, reusado tal cual via TransformerBottleneck):
      - FEM en el nivel 3 del encoder (c3, antes de pool3): self-attention
        no-local + gated convolution, para suprimir ruido de fondo y reforzar
        contexto antes del nivel mas profundo.
      - Coordinate Attention en los 4 skip connections: pooling direccional
        (fila/columna) en vez de un pooling global como CBAM.
      - Reverse Attention en la segunda etapa del decoder (dec2, resolucion
        simetrica a donde va FEM en el encoder): reemplaza la fusion directa
        del skip por una version pesada por el mapa de borde invertido del
        propio decoder.
    """
    def __init__(self, in_channels, num_classes,
                 transformer_depth=2, transformer_heads=8, transformer_dropout=0.1,
                 fem_heads=8):
        super().__init__()
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.fem = FeatureEnhancementModule(256, n_heads=fem_heads)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Cuello de botella: DoubleConv + atencion global (igual que TransUNet)
        self.bottleneck = DoubleConv(512, 1024, dropout=0.3)
        self.transformer = TransformerBottleneck(
            1024, depth=transformer_depth, num_heads=transformer_heads,
            dropout=transformer_dropout, base_grid=16)

        # Coordinate Attention en cada nivel de skip
        self.ca1 = CoordinateAttention(64)
        self.ca2 = CoordinateAttention(128)
        self.ca3 = CoordinateAttention(256)
        self.ca4 = CoordinateAttention(512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512, dropout=0.2)  # 512 + 512 (skip)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ram = ReverseAttention(256)                # segunda etapa: reverse attention en vez de concat directo
        self.dec2 = DoubleConv(512, 256)  # 256 (up2) + 256 (skip via RAM)
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
        c3 = self.fem(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        b = self.bottleneck(p4)
        b = self.transformer(b)

        u1 = self.up1(b)
        u1 = torch.cat([u1, self.ca4(c4)], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = self.ram(u2, self.ca3(c3))
        d2 = self.dec2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, self.ca2(c2)], dim=1)
        d3 = self.dec3(u3)

        u4 = self.up4(d3)
        u4 = torch.cat([u4, self.ca1(c1)], dim=1)
        d4 = self.dec4(u4)

        out = self.out_conv(d4)
        return out
