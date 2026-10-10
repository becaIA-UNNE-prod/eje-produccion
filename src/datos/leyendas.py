"""Leyendas MNC por campaña; no importa comandos ni datos raster."""
import json
from pathlib import Path
"""Leyendas oficiales MNC. Las mascaras conservan los codigos originales."""
import numpy as np

REMAP_2021 = np.zeros(256, dtype=np.uint8)
REMAP_2021[10], REMAP_2021[11], REMAP_2021[16] = 2, 3, 4
REMAP_2021[[12, 13, 14, 15, 17, 18, 19, 21, 22, 26, 28]] = 1
CODIGOS_2021 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 26, 28, 255}


def verificar_codigos(codigos, ignorar_codigo20=False):
    desconocidos = set(map(int, codigos)) - CODIGOS_2021
    if ignorar_codigo20:
        desconocidos.discard(20)
    if desconocidos:
        raise ValueError(f'Codigos MNC 2021 sin documentar: {sorted(desconocidos)}')

ROOT = Path(__file__).resolve().parents[2]
REMAP_1819 = np.zeros(256,dtype=np.uint8)
REMAP_1819[[0,3]], REMAP_1819[[1,2]], REMAP_1819[4] = 2,3,4
REMAP_1819[[5,6,7,8,9,11,12,13,14,15]] = 1
REMAP_2020 = np.zeros(256,dtype=np.uint8)
REMAP_2020[[10,11,16]] = [2,3,4]
REMAP_2020[[12,13,14,15,17,18,19,21,22]] = 1
LEGENDS = {'2018_2019':(REMAP_1819,'etiqueta_mnc1819_{}_10m.tif'),
           '2019_2020':(REMAP_2020,'etiqueta_mnc_{}_10m.tif'),
           '2020_2021':(REMAP_2021,'etiqueta_mnc2021_{}_10m.tif')}

def legend(year):
 if year=='2021_2022':
  cfg=json.loads((ROOT/'config/leyenda_mnc_2021_2022.json').read_text());return np.array(cfg['remap'],dtype=np.uint8),set(cfg['codigos_documentados'])|{255},255
 lut=LEGENDS[year][0].astype(np.uint8)
 if year=='2018_2019':return lut,{0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16},16
 if year=='2019_2020':return lut,{0}|set(range(10,20))|{21,22,255},255
 return lut,set(range(10,20))|{20,21,22,26,28,255},255

def check_codes(codes,allowed):
 unknown=set(map(int,codes))-allowed
 if unknown:raise ValueError(f'Codigos sin leyenda: {sorted(unknown)}')
