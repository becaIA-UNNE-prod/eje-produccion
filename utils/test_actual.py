"""Configuracion activa de TEST; el SPLIT antiguo conserva reproducibilidad."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/split_extension_este.json'
def cargar_config(ruta=CONFIG):
 cfg=json.loads(Path(ruta).read_text());seen=set()
 for role in ['train','val','test']:
  group=cfg[role]
  if not isinstance(group,list) or not group:raise ValueError(f'Split vacio: {role}')
  if len(set(group))!=len(group):raise ValueError(f'Tiles duplicados en {role}')
  if seen.intersection(group):raise ValueError('Tiles compartidos entre particiones')
  if any(not re.fullmatch(r'\d{2}[A-Z]{3}',t) for t in group):raise ValueError('Tile invalido')
  seen.update(group)
 return cfg

def preflight(cfg,campanas=None):
 base=Path(cfg['preparacion_base']);protocol=json.loads((base/'protocolo.json').read_text())
 for role in ['train','val']:
  if cfg[role]!=protocol['split_original'][role]:raise ValueError('Regenerar exclusiones: TRAIN/VAL difieren de la preparacion')
 prepared=json.loads((base/'preparacion_verificada.json').read_text())
 if cfg['test']!=prepared['tiles']:raise ValueError('TEST distinto del preparado')
 years=campanas or cfg['campanas']
 if not set(years)<=set(prepared['campanas']):raise ValueError('Campana sin preparacion')
 for year in years:
  for tile in cfg['test']:
   for scope in ['completa','independiente']:
    if not (base/'mascaras'/f'{year}_{tile}_{scope}.tif').exists():raise FileNotFoundError('Mascara no preparada')
 return base
