#!/usr/bin/env python3
"""Actualiza el catálogo leyendo solo cabeceras de composites en floryacy."""
import argparse
import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    import rasterio
    from rasterio.warp import transform
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida',type=Path,required=True)
    args=parser.parse_args()
    split=json.loads((ROOT/'config/split_extension_este.json').read_text())
    campaigns=json.loads((ROOT/'config/campanas.json').read_text())
    storage=json.loads((ROOT/'config/almacenamiento.json').read_text())
    features=[]
    for role in ('train','val','test','test_anterior_no_asignado'):
        for tile in split.get(role,[]):
            records=[]
            for year in split['campanas'] if role=='test' else campaigns['preparacion_train_val']:
                item=campaigns['campanas'][year]; extension=role=='test'
                source=Path(storage['rutas_operativas']['extension_este' if extension else 'multicampana'])/item['composites_extension' if extension else 'composites_multicampana']/f"{tile}_{item['meses'][0]}_median.tif"
                with rasterio.open(source) as raster:
                    # Esquinas reales de la grilla, también si existiera rotación.
                    corners=[raster.transform*p for p in [(0,0),(raster.width,0),(raster.width,raster.height),(0,raster.height),(0,0)]]
                    xs,ys=transform(raster.crs,'EPSG:4326',[p[0] for p in corners],[p[1] for p in corners])
                    records.append(dict(campana=year,ruta=str(source),crs=str(raster.crs),shape=[raster.height,raster.width],coordinates=list(map(list,zip(xs,ys)))))
            features.append(dict(type='Feature',properties=dict(tile=tile,geometrias_por_campana=records),geometry=dict(type='Polygon',coordinates=[records[0]['coordinates']])))
    from utils.corridas import guardar_json
    guardar_json(args.salida,dict(type='FeatureCollection',medido_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nota='Cabeceras del primer mes de cada campaña; contornos completos, no parches válidos.',features=features))
    print(f'{len(features)} tiles: {args.salida}')
if __name__=='__main__':
    main()
