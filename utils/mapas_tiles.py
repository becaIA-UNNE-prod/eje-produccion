"""Mapas livianos y autónomos; no abren rasters ni requieren librerías GIS."""
import hashlib
import json
from pathlib import Path


def generar_mapa(config_dir, salida, titulo='Tiles del split vigente'):
    config_dir, salida = Path(config_dir), Path(salida)
    split_path = config_dir / 'split_extension_este.json'
    split = json.loads(split_path.read_text())
    catalog = json.loads((config_dir / 'geometria_tiles.json').read_text())
    campaigns = json.loads((config_dir / 'campanas.json').read_text())
    roles = {}
    for role in ('train', 'val', 'test'):
        for tile in split[role]:
            if tile in roles:
                raise ValueError('Tile repetido entre roles: ' + tile)
            roles[tile] = role
    available = {f['properties']['tile'] for f in catalog['features']}
    if set(roles) - available:
        raise ValueError('Faltan contornos: ' + ', '.join(sorted(set(roles) - available)))
    features = []
    for feature in catalog['features']:
        props = dict(feature['properties'])
        props['rol'] = roles.get(props['tile'], 'inactivo')
        props['campanas_usadas'] = split['campanas'] if props['rol'] == 'test' else campaigns['preparacion_train_val'] if props['rol'] in ('train', 'val') else []
        for year in props['campanas_usadas']:
            if year not in {g['campana'] for g in props['geometrias_por_campana']}:
                raise ValueError(f"Falta contorno de {props['tile']} / {year}")
        features.append(dict(feature, properties=props))
    data = dict(type='FeatureCollection', features=features)
    metadata = dict(titulo=titulo, split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    catalogo_sha256=hashlib.sha256((config_dir/'geometria_tiles.json').read_bytes()).hexdigest(),
                    conteos={role: len(split[role]) for role in ('train','val','test')},
                    alcance='Contornos completos de composites, no cobertura de parches ni máscara de exclusión.')
    salida.mkdir(parents=True, exist_ok=True)
    (salida/'tiles.geojson').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (salida/'mapa.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    template = (Path(__file__).parent/'mapa_tiles.html').read_text()
    payload = json.dumps(dict(datos=data,meta=metadata,campanas=split['campanas']),ensure_ascii=False).replace('<','\\u003c')
    (salida/'mapa_tiles.html').write_text(template.replace('__DATOS__',payload))
    return metadata


def actualizar_indice(base):
    """Índice portátil de mapas versionados, independiente de /mnt."""
    import html
    base = Path(base)
    links = [('vigente/mapa_tiles.html', 'Split vigente')]
    links += [(str(p.relative_to(base)), p.parent.name)
              for p in sorted((base/'corridas').glob('*/mapa_tiles.html'), reverse=True)]
    items = ''.join('<li><a href="'+html.escape(path, quote=True)+'">'+html.escape(label)+'</a></li>' for path,label in links)
    (base/'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Mapas de tiles</title><style>body{font:18px system-ui;max-width:900px;margin:50px auto;padding:20px;background:#f4f6fa;color:#182739}li{margin:16px 0}a{color:#1265ab}</style><h1>Mapas de tiles</h1><ul>'+items+'</ul><p>Mapas autónomos del split vigente y de la configuración al iniciar cada corrida. Se distribuyen con Git. Contornos completos, no parches válidos.</p></html>')
