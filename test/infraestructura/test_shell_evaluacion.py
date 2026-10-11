"""Coordinación en réplica /tmp; intérprete simulado y locks locales."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]


def run_shell(entry='canonical',scenario='',baseline=None):
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);repo=base/'repo con espacios';logs=base/'logs';logs.mkdir()
        out=base/'salida con espacios';out.mkdir();exp=base/'exp con espacios'
        if baseline:
            actual=repo/'src/inferencia_multicampana.sh';actual.parent.mkdir(parents=True)
            actual.write_text((baseline/'src/inferencia_multicampana.sh').read_text().replace('/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana',str(logs)))
        else:
            for name in ['scripts/evaluacion/inferencia_multicampana.sh','src/inferencia_multicampana.sh']:
                p=repo/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text((ROOT/name).read_text().replace('/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana',str(logs)))
            actual=repo/('src/inferencia_multicampana.sh' if entry=='wrapper' else 'scripts/evaluacion/inferencia_multicampana.sh')
        helper = repo/'scripts/lib/corridas.sh'
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text((ROOT/'scripts/lib/corridas.sh').read_text())
        binpath=base/'bin';binpath.mkdir();fake=binpath/'python_simulado'
        fake.write_text('#!'+sys.executable+'''\nimport json,os,sys
from pathlib import Path
os.fstat(9)
Path(os.environ['FIXTURE_TRACE']).write_text(json.dumps({'args':sys.argv[1:],'cwd':str(Path.cwd()),'lock_abierto':True}))
print('EVALUACION SIMULADA')
raise SystemExit(int(os.environ['FIXTURE_EXIT']))
''');fake.chmod(0o755)
        date=binpath/'date';date.write_text('#!/bin/sh\nprintf "%s\\n" FECHA_FIXTURE\n');date.chmod(0o755)
        trace=base/'trace.json'
        env=dict(os.environ,PY=str(fake),PATH=str(binpath)+os.pathsep+os.environ['PATH'],FIXTURE_TRACE=str(trace),FIXTURE_EXIT='7' if scenario=='fallo' else '0')
        env.update(CORDOBA_RUN_DIR=str(out), CORDOBA_RUNTIME_ROOT=str(base/'runtime'))
        argv=['bash',str(actual),str(exp),str(out)]
        if scenario=='faltan_args':argv=argv[:2]
        if scenario=='lock':
            with open(logs/'inferencia.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run(argv,cwd=base,env=env,capture_output=True,text=True)
        else:result=subprocess.run(argv,cwd=base,env=env,capture_output=True,text=True)
        event=json.loads(trace.read_text()) if trace.exists() else None
        state={p.name:p.read_text() for p in out.iterdir() if p.is_file()}
        if 'inferencia.pid' in state:
            assert state['inferencia.pid'].strip().isdigit();state['inferencia.pid']='PID_FIXTURE\n'
        if event:assert event['cwd']==str(repo)
        value={'rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'event':event,'state':state}
        return json.loads(json.dumps(value).replace(str(base),'FIXTURE_ROOT'))


class ShellEvaluacionTest(unittest.TestCase):
    def test_wrapper_y_destino_equivalentes_con_lock_heredado(self):
        actual=run_shell();self.assertEqual(actual,run_shell('wrapper'))
        self.assertEqual(actual['rc'],0,actual['stderr'])
        self.assertEqual(actual['event']['args'][:3],['-u','-m','src.evaluacion.postprocesar_multicampana'])
        self.assertEqual(actual['event']['args'][-1],'--lock-adquirido')
        self.assertEqual(actual['state']['inferencia.exitcode'],'0\n')
        self.assertTrue(actual['event']['lock_abierto'])
        self.assertEqual(actual['state']['inicio.txt'],'FECHA_FIXTURE\n')
        self.assertEqual(actual['state']['fin.txt'],'FECHA_FIXTURE\n')

    def test_fallo_y_lock_ocupado(self):
        failed=run_shell(scenario='fallo');self.assertEqual(failed['rc'],7)
        self.assertEqual(failed['state']['inferencia.exitcode'],'7\n')
        locked=run_shell(scenario='lock');self.assertEqual(locked['rc'],1)
        self.assertIsNone(locked['event']);self.assertEqual(locked['state'],{})

    def test_faltan_argumentos_no_ejecuta_evaluacion(self):
        result=run_shell(scenario='faltan_args')
        self.assertNotEqual(result['rc'],0);self.assertIsNone(result['event'])
