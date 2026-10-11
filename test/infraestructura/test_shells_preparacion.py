"""Shells en réplica /tmp con Python simulado: no red, GPU ni datos reales."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
NAMES=['descargar_2020_2021_scl','rehacer_2018_2019_scl','descargar_extension_este','preparar_extension_este']
V3='/mnt/yacy_3/prod_bis/ferreyra'
V1='/mnt/yacy_1/prod/ferreyra/extension_este_multicampana'
FAKE_PY='''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
program=sys.stdin.read() if args[:2]==['-u','-'] else ''
record={'args':args,'cwd':str(Path.cwd()),'stdin':program}
with open(os.environ['FIXTURE_TRACE'],'a') as stream:
    stream.write(json.dumps(record)+'\\n')
scenario=os.environ.get('FIXTURE_SCENARIO','')
if scenario=='reserva' and ('disk_usage' in program):
    raise SystemExit(9)
if scenario=='campana2020' and 'buscar' in args and '--inicio' in args and args[args.index('--inicio')+1]=='2020-12-01':
    raise SystemExit(8)
if scenario=='evaluacion' and 'src.evaluacion.evaluar_extension' in args:
    raise SystemExit(7)
if scenario=='reintento' and 'descargar' in args and '--bandas' in args:
    marker=Path(os.environ['FIXTURE_TRACE']+'.reintento')
    if not marker.exists():
        marker.write_text('fallo simulado')
        raise SystemExit(6)
print('SIMULADO')
'''


def run_flow(name,entry='canonical',scenario='',baseline=None):
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp)
        repo=base/'repo con espacios'
        vol3=base/'vol3'; vol3.mkdir()
        vol1=base/'vol1'; vol1.mkdir()
        out=base/'salida con espacios'
        (vol3/'logs_multicampana').mkdir()
        (vol3/'tiles.txt').write_text('20AAA\n')
        for campaign in ['2018_2019','2020_2021']:
            (vol3/('sentinel2_cordoba_'+campaign)).mkdir()
        if scenario=='cache':
            (vol3/'sentinel2_cordoba_2020_2021/productos.json').write_text('[]')
        def relocate(body):
            return body.replace(V1,str(vol1)).replace(V3,str(vol3))
        if baseline is not None:
            actual=repo/'src'/f'{name}.sh'
            actual.parent.mkdir(parents=True)
            actual.write_text(relocate((baseline/'src'/f'{name}.sh').read_text()))
        else:
            for relative in ['scripts/preparacion/'+name+'.sh','src/'+name+'.sh']:
                path=repo/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(relocate((ROOT/relative).read_text()))
            actual=repo/('src' if entry=='wrapper' else 'scripts/preparacion')/f'{name}.sh'
        helper = repo/'scripts/lib/corridas.sh'
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text((ROOT/'scripts/lib/corridas.sh').read_text())
        binpath=base/'bin'; binpath.mkdir()
        fake=binpath/'python_simulado'
        fake.write_text('#!'+sys.executable+'\n'+FAKE_PY)
        fake.chmod(0o755)
        date=binpath/'date'
        date.write_text('#!/bin/sh\nprintf "%s\\n" FECHA_FIXTURE\n')
        date.chmod(0o755)
        trace=base/'trace.jsonl'
        env=dict(os.environ,PY=str(fake),PATH=str(binpath)+os.pathsep+os.environ['PATH'],
                 FIXTURE_TRACE=str(trace),FIXTURE_SCENARIO=scenario)
        env.update(CORDOBA_RUN_DIR=str(out), CORDOBA_RUNTIME_ROOT=str(base/'runtime'))
        command=['bash',str(actual)]
        if name in ['descargar_extension_este','preparar_extension_este']:
            command.append(str(out))
        if scenario=='lock':
            with open(vol1/('descarga.lock' if name=='descargar_extension_este' else 'preparacion.lock'),'a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run(command,cwd=base,env=env,capture_output=True,text=True)
        else:
            result=subprocess.run(command,cwd=base,env=env,capture_output=True,text=True)
        events=[json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        for event in events:
            assert event['cwd']==str(repo),event
        state={}
        if out.exists():
            for file in out.iterdir():
                if file.is_file():
                    text=file.read_text()
                    if file.suffix=='.pid':
                        assert text.strip().isdigit(),text
                        text='PID_FIXTURE\n'
                    state[file.name]=text
        value={'rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'events':events,'state':state}
        # Only fixture roots differ between separate replicas.
        return json.loads(json.dumps(value).replace(str(base),'FIXTURE_ROOT'))


def modules(result):
    return [event['args'] for event in result['events'] if '-m' in event['args']]


class ShellsPreparacionTest(unittest.TestCase):
    def test_cuatro_entradas_equivalentes_desde_otro_cwd(self):
        for name in NAMES:
            with self.subTest(name=name):
                actual=run_flow(name)
                anterior=run_flow(name,entry='wrapper')
                self.assertEqual(actual,anterior)
                self.assertEqual(actual['rc'],0,actual['stderr'])
                self.assertTrue(actual['events'])
                if name=='descargar_extension_este':
                    self.assertEqual(actual['state']['descarga.exitcode'],'0\n')
                    for year in ['2018_2019','2019_2020','2020_2021','2021_2022']:
                        self.assertEqual(actual['state'][year+'.exitcode'],'0\n')
                    self.assertEqual(len(modules(actual)),24)
                    self.assertFalse(any('componer' in args for args in modules(actual)))
                elif name=='preparar_extension_este':
                    self.assertEqual(actual['state']['preparacion.exitcode'],'0\n')
                    self.assertEqual(sum('componer' in args for args in modules(actual)),4)
                    stages=[a[a.index('-m')+1] for a in modules(actual)[-2:]]
                    self.assertEqual(stages,['src.datos.preparar_extension','src.evaluacion.evaluar_extension'])

    def test_campana2020_orden_cache_y_reintento(self):
        actual=run_flow('descargar_2020_2021_scl')
        args=modules(actual)
        self.assertEqual([a[a.index('-m')+2] for a in args],['buscar','descargar','seleccionar','descargar','verificar','componer'])
        self.assertEqual(args[0][args[0].index('--max-nodata')+1],'100')
        selected=args[2][args[2].index('--meses')+1:]
        self.assertEqual(selected,['202012','202101','202102','202103','202104','202105'])
        cache=run_flow('descargar_2020_2021_scl',scenario='cache')
        self.assertFalse(any('buscar' in a for a in modules(cache)))
        retry=run_flow('descargar_2020_2021_scl',scenario='reintento')
        self.assertEqual(retry['rc'],0)
        self.assertEqual(sum('--bandas' in a for a in modules(retry)),2)

    def test_rehacer_preserva_inferencia_evaluacion_y_graficos(self):
        actual=run_flow('rehacer_2018_2019_scl')
        args=[event['args'] for event in actual['events']]
        self.assertEqual([a[a.index('-m')+2] for a in modules(actual) if a[a.index('-m')+1].startswith('src.datos.')],['descargar','seleccionar','descargar','verificar','componer'])
        inference=[a for a in args if 'src.inferencia.inferir_tile' in a]
        self.assertEqual([a[a.index('--modelo')+1] for a in inference],['unet','fcrtransunet'])
        self.assertEqual(args[-2][:4],['-m','src.historico.evaluar_contra_mnc','--leyenda','1819'])
        self.assertEqual(args[-1],['-m','src.historico.comparar_con_1718','--comparacion','scl'])

    def test_extension_continua_otras_campanas_y_registra_fallo(self):
        actual=run_flow('descargar_extension_este',scenario='campana2020')
        self.assertEqual(actual['rc'],1)
        self.assertEqual(actual['state']['2020_2021.exitcode'],'8\n')
        self.assertEqual(actual['state']['2021_2022.exitcode'],'0\n')
        self.assertEqual(actual['state']['descarga.exitcode'],'1\n')
        self.assertIn('2021-12-01',str(modules(actual)))

    def test_locks_y_reserva_en_replica(self):
        for name in ['descargar_extension_este','preparar_extension_este']:
            with self.subTest(name=name):
                locked=run_flow(name,scenario='lock')
                self.assertEqual(locked['rc'],1)
                self.assertEqual(locked['events'],[])
                self.assertFalse(any(file.endswith('.pid') for file in locked['state']))
                low=run_flow(name,scenario='reserva')
                self.assertNotEqual(low['rc'],0)
                self.assertEqual(modules(low),[])

    def test_preparar_extension_preserva_fallo_evaluacion(self):
        actual=run_flow('preparar_extension_este',scenario='evaluacion')
        self.assertEqual(actual['rc'],7)
        self.assertEqual(actual['state']['preparacion.exitcode'],'7\n')
        self.assertEqual(actual['events'][-1]['args'][2],'src.evaluacion.evaluar_extension')
