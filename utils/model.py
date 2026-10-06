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
