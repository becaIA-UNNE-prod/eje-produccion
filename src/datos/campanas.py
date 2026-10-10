"""Catalogo dic-may y rutas acreditadas; no abre datasets ni comandos."""
import json
import re
from pathlib import Path
from utils.rutas import ROOT, MULTICAMPANA_ROOT, EXTENSION_ROOT, MNC_ROOT

CONFIG = ROOT / 'config/campanas.json'


def cargar_config(ruta=CONFIG):
    cfg = json.loads(Path(ruta).read_text())
    if cfg['bandas'] != ['B02', 'B03', 'B04', 'B08'] or cfg['normalizacion'] != 10000:
        raise ValueError('Contrato dic-may incompatible: bandas/normalizacion')
    for year, item in cfg['campanas'].items():
        if not re.fullmatch(r'\d{4}_\d{4}', year) or int(year[5:]) != int(year[:4]) + 1:
            raise ValueError('Campana no consecutiva: ' + year)
        expected = [year[:4] + '12'] + [year[5:] + f'{m:02}' for m in range(1, 6)]
        if item['meses'] != expected:
            raise ValueError('Ventana dic-may incompatible: ' + year)
        for key in ('composites_multicampana', 'composites_extension', 'fuente_mnc'):
            path = Path(item[key])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Ruta de campana debe ser relativa a su base: ' + key)
    for year in cfg['preparacion_train_val']:
        if not cfg['campanas'][year]['dataset']:
            raise ValueError('Campana de preparacion sin dataset: ' + year)
    return cfg


CATALOGO = cargar_config()
BANDAS = tuple(CATALOGO['bandas'])
MASCARAS = ROOT / 'mascaras_procesadas'  # Alias acreditado, bytes externos.
LOGS = MULTICAMPANA_ROOT / 'logs_multicampana'


def meses(year):
    return list(CATALOGO['campanas'][year]['meses'])


def composites(year, tile, extension=False, base=None):
    key = 'composites_extension' if extension else 'composites_multicampana'
    base = Path(base) if base is not None else (EXTENSION_ROOT if extension else MULTICAMPANA_ROOT)
    directory = base / CATALOGO['campanas'][year][key]
    return [directory / f'{tile}_{month}_median.tif' for month in meses(year)]


def files(year, tile):
    return composites(year, tile, extension=True)


def fuentes_mnc(years):
    return {year: MNC_ROOT / CATALOGO['campanas'][year]['fuente_mnc'] for year in years}


def campanas_preparacion():
    return [(year, MULTICAMPANA_ROOT / item['composites_multicampana'],
             list(item['meses']), item['leyenda'], item['patron_mascara'],
             MULTICAMPANA_ROOT / 'dataset' / item['dataset'])
            for year in CATALOGO['preparacion_train_val']
            for item in [CATALOGO['campanas'][year]]]
