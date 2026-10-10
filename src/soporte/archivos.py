"""Escritura JSON atomica y checksum, sin rutas ni comandos propios."""
import hashlib
import json

def guardar(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for data in iter(lambda:file.read(8*1024**2),b''):
            digest.update(data)
    return digest.hexdigest()
