"""Catalogo dic-may y rutas acreditadas; no abre datasets ni comandos."""
import json
import copy
import os
import re
from pathlib import Path
from utils.rutas import ROOT, MULTICAMPANA_ROOT, EXTENSION_ROOT, MNC_ROOT

CONFIG = ROOT / 'config/campanas.json'


def cargar_config(ruta=CONFIG):
    cfg = json.loads(Path(ruta).read_text())
    if cfg['ventana'] != 'diciembre-mayo':
        raise ValueError('Ventana del catalogo incompatible')
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
        dataset = item['dataset']
        if dataset is not None:
            if not isinstance(dataset, str) or not dataset or Path(dataset).is_absolute() or len(Path(dataset).parts) != 1 or dataset in ('.', '..'):
                raise ValueError('Dataset debe ser un nombre relativo: ' + year)
        pattern = item['patron_mascara']
        if not isinstance(pattern, str) or pattern.count('{}') != 1 or '{' in pattern.replace('{}', '') or '}' in pattern.replace('{}', '') or '/' in pattern or '\\' in pattern:
            raise ValueError('Patron de mascara incompatible: ' + year)
        if item['leyenda'] not in ('1819', '2020', '2021', '2022'):
            raise ValueError('Leyenda no soportada: ' + year)
    def validar_lista(years, label):
        if not isinstance(years, list) or not years or any(not isinstance(y, str) for y in years) or len(set(years)) != len(years) or not set(years) <= set(cfg['campanas']):
            raise ValueError('Lista de campanas incompatible: ' + label)
    validar_lista(cfg['preparacion_train_val'], 'preparacion_train_val')
    for name, protocol in cfg['protocolos'].items():
        validar_lista(protocol['campanas'], name)
        validar_lista(protocol['campanas_exclusion'], name + '/exclusion')
        if not set(protocol['campanas_exclusion']) <= set(cfg['preparacion_train_val']):
            raise ValueError('Exclusion requiere campanas de preparacion: ' + name)
        tiles = protocol['tiles']
        if not isinstance(tiles, list) or not tiles or any(not isinstance(t, str) or not re.fullmatch(r'\d{2}[A-Z]{3}', t) for t in tiles) or len(set(tiles)) != len(tiles):
            raise ValueError('Tiles del protocolo incompatibles: ' + name)
    for year in cfg['preparacion_train_val']:
        if not cfg['campanas'][year]['dataset']:
            raise ValueError('Campana de preparacion sin dataset: ' + year)
    return cfg


CATALOGO = cargar_config()
BANDAS = tuple(CATALOGO['bandas'])
NORMALIZACION = CATALOGO['normalizacion']
MASCARAS = ROOT / 'mascaras_procesadas'  # Alias acreditado, bytes externos.
LOGS = Path(os.environ['CORDOBA_INVENTARIOS_DIR']) if 'CORDOBA_INVENTARIOS_DIR' in os.environ else MULTICAMPANA_ROOT / 'logs_multicampana'


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


def protocolo(nombre):
    """Alcance fijo de un flujo; no sustituye config/split_extension_este.json."""
    return copy.deepcopy(CATALOGO['protocolos'][nombre])
