"""Compatibilidad: las implementaciones viven en model/.

Conservar esta ruta para consumidores y objetos serializados historicos.
"""
from model import (DoubleConv, SimpleUNet, TransformerBottleneck, TransUNet,
                   LTAE2d, UTAE, SelfAttentionConv, GatedConv,
                   FeatureEnhancementModule, CoordinateAttention, ReverseAttention,
                   FCRTransUNet)
from model.utae import _upsample_attn, _reponderar_temporal
