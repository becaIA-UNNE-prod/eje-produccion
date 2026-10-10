"""Regenerar figuras desde predicciones existentes, sin inferencia nueva."""
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.datos.campanas import files
from src.figuras.test_actual import figura_extension
def crear_parser():
 p=argparse.ArgumentParser();p.add_argument('--salida',type=Path,required=True);p.add_argument('--esperar',action='store_true');return p

def main(argv=None):
 a=crear_parser().parse_args(argv);out=a.salida
 try:
  if a.esperar:
   for attempt in range(1080):
    code=out/'inferencia.exitcode'
    if code.exists():
     if code.read_text().strip()!='0':raise RuntimeError('Inferencia termino con error')
     break
    time.sleep(10)
   else:raise TimeoutError('Inferencia pendiente tras 3 horas')
  protocol=json.loads((out/'protocolo.json').read_text());base=Path(protocol['preparacion_base']);cfg=protocol['split']
  for name,record in protocol['modelos'].items():
   modelout=out/'modelos'/name
   for year in cfg['campanas']:
    for tile in cfg['test']:figura_extension(modelout,year,tile,modelout/'predicciones'/year/f'prediccion_{tile}.tif',files(year,tile)[2],maskbase=base,title=record['nombre'])
  (out/'figuras.exitcode').write_text('0\n');print('Figuras verificadas y actualizadas',flush=True)
 except Exception:
  (out/'figuras.exitcode').write_text('1\n');raise

if __name__=='__main__':
 main()
