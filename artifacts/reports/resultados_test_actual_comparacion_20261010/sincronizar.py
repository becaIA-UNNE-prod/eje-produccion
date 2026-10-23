import pathlib,subprocess,time,tarfile,json,datetime
local=pathlib.Path(__file__).resolve().parent
remote='/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/comparacion_test_actual_20261010T092436'
for attempt in range(2880):
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','floryacy',f'cat {remote}/inferencia.exitcode {remote}/figuras.exitcode 2>/dev/null'],capture_output=True,text=True,timeout=45)
 codes=result.stdout.split()
 if any(code!='0' for code in codes):raise RuntimeError('Error remoto; revisar inferencia.log o figuras.log')
 if result.returncode==0 and codes==['0','0']:break
 time.sleep(60)
else:raise TimeoutError('Evaluacion pendiente tras 48 horas')
archive=local/'resultados.tar.gz'
with archive.open('wb') as dst:
 subprocess.run(['ssh','floryacy',f"cd {remote} && tar -czf - --exclude='*.pth' --exclude='predicciones' figuras modelos protocolo.json evaluacion_test_actual.json metricas_tiles.json comparacion_metricas.csv README.md fin.txt"],stdout=dst,check=True)
with tarfile.open(archive) as tf:
 for member in tf.getmembers():
  path=pathlib.PurePosixPath(member.name)
  if not member.isfile() or path.is_absolute() or '..' in path.parts:continue
  target=local/path;target.parent.mkdir(parents=True,exist_ok=True)
  with tf.extractfile(member) as src,target.open('wb') as dst:dst.write(src.read())
(local/'estado_transferencia.json').write_text(json.dumps({'estado':'completo','fecha':datetime.datetime.now().isoformat(),'remoto':remote},indent=2)+'\n')
print('Resultados de los tres modelos copiados a '+str(local),flush=True)
