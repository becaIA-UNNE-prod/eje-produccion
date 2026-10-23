"""Organiza el reporte técnico existente por superficie y campañas, sin inferencia."""
import argparse
import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CONDICIONES = [
    ('reducida_2020_2021', '3 tiles · 1 campaña', '3-tiles-1-campana', 3, 1),
    ('reducida_tres', '3 tiles · 3 campañas', '3-tiles-3-campanas', 3, 3),
    ('amplia_2020_2021', '14 tiles · 1 campaña', '14-tiles-1-campana', 14, 1),
    ('amplia_tres', '14 tiles · 3 campañas', '14-tiles-3-campanas', 14, 3),
]


def organizar(reporte):
    filas = json.loads((reporte / 'metricas/resumen_2021_2022.json').read_text())
    figuras = sorted((reporte / 'figuras').glob('*.png'))
    instante = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    respaldo = reporte / 'versiones' / instante
    respaldo.mkdir(parents=True)
    for nombre in ['index.html', 'RESULTADOS.md', 'sha256.json']:
        if (reporte / nombre).exists():
            shutil.copy2(reporte / nombre, respaldo / nombre)

    def tabla(seleccion):
        campos = [('accuracy', 'Exactitud global'), ('miou', 'mIoU'),
                  ('f1_macro', 'F1 macro'), ('f1_Mani', 'F1 maní'),
                  ('precision_mani', 'Precisión maní'), ('recall_mani', 'Recall maní')]
        cabecera = '<tr><th>Modelo</th>' + ''.join(f'<th>{t} (%)</th>' for _, t in campos) + '</tr>'
        cuerpo = ''.join('<tr><td>' + ('U-Net' if r['arquitectura'] == 'unet' else 'Transformer puro (Segmenter)') + '</td>' + ''.join('<td>' + f'{r[k]:.2f}'.replace('.', ',') + '</td>' for k, _ in campos) + '</tr>' for r in seleccion)
        return '<div class="scroll"><table><thead>' + cabecera + '</thead><tbody>' + cuerpo + '</tbody></table></div>'

    def galeria(titulo, archivos):
        items = []
        for f in archivos:
            tipo = ('Matriz de confusión' if f.name.startswith('confusion_') else
                    'Detalle de maní' if f.name.startswith('detalle_') else
                    'Panorama de comparación' if f.name.startswith('panorama_') else 'Curvas de validación')
            modelo = 'U-Net' if '_unet_' in f.stem else 'Transformer puro' if '_segmenter_' in f.stem else 'U-Net y Transformer puro'
            anio = next((f'{y}/{y+1}' for y in range(2018, 2022) if f'{y}_{y+1}' in f.stem), '')
            # 2020_2021 puede estar en la condición: el año TEST ocupa el sufijo.
            for y in range(2018, 2022):
                if f.stem.endswith(f'_{y}_{y+1}') or any(f.stem.endswith(f'_{y}_{y+1}_{t}') for t in ['20HNJ', '20HNH']):
                    anio = f'{y}/{y+1}'
            tile = next((t for t in ['20HNJ', '20HNH'] if t in f.stem), '')
            nombre = ' · '.join(x for x in [tipo, modelo, ('TEST ' + anio) if anio else '', tile] if x)
            ruta = 'figuras/' + f.name
            items.append(f'<figure><figcaption>{html.escape(nombre)}</figcaption><a href="{ruta}"><img loading="lazy" src="{ruta}" alt="{html.escape(nombre)}"></a><details class="origen"><summary>Identificador original</summary><code>{html.escape(f.name)}</code></details></figure>')
        return f'<details class="galeria"><summary>{titulo} ({len(archivos)})</summary>' + ''.join(items) + '</details>'

    nav = '<nav aria-label="Sectores del ensayo">' + ''.join(f'<a href="#{ident}">{titulo}</a>' for _, titulo, ident, _, _ in CONDICIONES) + '</nav>'
    sectores = []
    md = ['# Resultados por superficie y campañas', '',
          'Cuatro condiciones con 3.072 parches cada una. Cada condición compara U-Net y Transformer puro. La cantidad de tiles indica la extensión geográfica de procedencia, no un aumento del área total muestreada.', '']
    asignadas = []
    for condicion, titulo, ident, tiles, campanas in CONDICIONES:
        seleccion = [r for r in filas if r['condicion'] == condicion]
        assert len(seleccion) == 2
        entrenamiento = '2020/2021, 3.072 parches' if campanas == 1 else '2018/2019, 2019/2020 y 2020/2021, 1.024 parches por campaña'
        territorio = '20HLG, 20HLH y 20HMG' if tiles == 3 else '13 tiles del TRAIN original más 20HLJ, incorporado para reforzar el núcleo manisero de Córdoba'
        propias = [f for f in figuras if f'_{condicion}_' in f.name or f.stem == f'curvas_{condicion}']
        asignadas.extend(propias)
        principal = [f for f in propias if '2021_2022' in f.stem]
        historicas = [f for f in propias if f not in principal and not f.name.startswith('curvas_')]
        curvas = [f for f in propias if f.name.startswith('curvas_')]
        sectores.append(f'<section id="{ident}"><h2>{titulo}</h2><div class="definicion"><p><strong>Superficie de procedencia:</strong> {territorio}.</p><p><strong>Campañas de entrenamiento:</strong> {entrenamiento}.</p><p><strong>Total por modelo:</strong> 3.072 parches, 100 épocas. Misma selección para ambas arquitecturas.</p></div><h3>Resultado principal: TEST 2021/2022</h3><p>Campaña ausente de TRAIN y VAL. Métricas agregadas de 20HNJ y 20HNH, criterio independiente.</p>{tabla(seleccion)}{galeria("Mapas y matrices del TEST 2021/2022", principal)}{galeria("Curvas de entrenamiento y validación", curvas)}{galeria("Anexo: TEST de las campañas 2018/2019, 2019/2020 y 2020/2021", historicas)}<a href="#inicio">Volver al índice de sectores</a></section>')
        md += [f'## {titulo}', '', f'- Superficie de procedencia: {territorio}.', f'- Campañas de entrenamiento: {entrenamiento}.', '- Total: 3.072 parches por modelo.', '',
               '| Modelo | mIoU (%) | F1 maní (%) | Precisión maní (%) | Recall maní (%) |', '|---|---:|---:|---:|---:|']
        for r in seleccion:
            md.append('| ' + ('U-Net' if r['arquitectura'] == 'unet' else 'Transformer puro') + ' | ' + ' | '.join(f'{r[k]:.2f}'.replace('.', ',') for k in ['miou','f1_Mani','precision_mani','recall_mani']) + ' |')
        md += ['', f'[Abrir mapas, curvas y matrices de este sector](index.html#{ident}).', '']
    assert len(asignadas) == len(set(asignadas)) == 76
    resumen = [f for f in figuras if f not in asignadas]
    assert len(resumen) == 3
    css = '''body{font:17px/1.5 system-ui,sans-serif;margin:0 auto;padding:28px;max-width:1450px;color:#193246;background:#f5f7f8}h1{font-size:36px}h2{font-size:30px;margin:0 0 18px}h3{font-size:23px}nav{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}nav a{padding:18px;background:#163e5c;color:white;text-align:center;font-weight:700;text-decoration:none}section{background:white;padding:30px;margin:32px 0;border-top:6px solid #2463a6;scroll-margin-top:20px}.definicion{background:#eef4f7;padding:12px 20px}.definicion p{margin:8px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:16px}th,td{padding:13px 10px;border-bottom:1px solid #cbd7df;text-align:left}th{background:#eaf0f5}summary{cursor:pointer;font-weight:600}.galeria>summary{padding:18px;background:#eef2f5;margin:15px 0}figure{margin:24px 0}figcaption{font-size:20px;font-weight:600;padding:12px 0}img{max-width:100%;height:auto}.origen{font-size:13px;color:#596c78}a{color:#165f93}.aviso{border-left:5px solid #d77b28;padding:12px 20px;background:#fff}footer{font-size:15px}@media(max-width:800px){nav{grid-template-columns:1fr 1fr}body{padding:14px}section{padding:18px}h1{font-size:28px}h2{font-size:25px}}'''
    pagina = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Archivo técnico: superficie vs. campañas</title><style>' + css + '</style><header id="inicio"><h1>Archivo técnico: superficie vs. campañas</h1><p>Elegí una de las cuatro condiciones. Cada sector reúne ambos modelos y todas sus evidencias.</p>' + nav + '<p class="aviso"><strong>Cómo comparar:</strong> para estudiar campañas, comparar 1 y 3 campañas manteniendo los tiles. Para estudiar superficie, comparar 3 y 14 tiles manteniendo las campañas. Siempre son 3.072 parches.</p><p><strong>Atención:</strong> campañas de entrenamiento y campañas evaluadas en TEST son conceptos distintos. El resultado principal de todos los sectores es TEST 2021/2022. Las demás campañas de TEST quedan en un anexo dentro de cada sector.</p><p><a href="RESULTADOS.md">Informe escrito por sectores</a> · <a href="PROTOCOLO_Y_RESULTADOS.md">Protocolo y limitaciones</a> · <a href="metricas/todas_las_metricas.csv">Métricas completas CSV</a></p></header>' + ''.join(sectores) + '<section id="comparacion"><h2>Comparación transversal de las cuatro condiciones</h2><p>Los gráficos originales conservan sus identificadores históricos: reducida corresponde a 3 tiles y amplia a 14 tiles; una corresponde a 1 campaña y tres a 3 campañas de entrenamiento.</p>' + galeria('Gráficos de comparación general', resumen) + '</section><footer><p>Referencia MNC, métricas por píxel. Una semilla por condición. No se demostró convergencia equivalente ni superioridad general de arquitectura. TEST usado en trabajos previos. No se modificaron métricas, predicciones ni figuras al reorganizar este archivo.</p></footer></html>'
    (reporte / 'index.html').write_text(pagina)
    anterior = (respaldo / 'RESULTADOS.md').read_text()
    marcador = '\n<!-- INFORME_TECNICO_PREVIO -->\n'
    anterior = anterior.split(marcador, 1)[-1] if marcador in anterior else anterior
    (reporte / 'RESULTADOS.md').write_text('\n'.join(md) + marcador + '\n## Análisis técnico y controles originales\n\n' + anterior)
    shutil.copy2(__file__, reporte / 'organizar_archivo.py')
    manifest = {str(f.relative_to(reporte)): hashlib.sha256(f.read_bytes()).hexdigest() for f in reporte.rglob('*') if f.is_file() and f != reporte / 'sha256.json'}
    (reporte / 'sha256.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'sectores':4, 'figuras_por_sector':19, 'figuras_transversales':3, 'respaldo':str(respaldo)}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reporte', type=Path, required=True)
    organizar(p.parse_args().reporte)
