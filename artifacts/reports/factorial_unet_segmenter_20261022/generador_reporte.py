"""Reporte del factorial: métricas, contrastes descriptivos y figuras pareadas.

No selecciona checkpoints ni ejecuta inferencia. Lee mapas ya evaluados.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.datos.campanas import files
from src.figuras.comunes import COLORS,NAMES

CONDITIONS=['reducida_2020_2021','reducida_tres','amplia_2020_2021','amplia_tres']
LABELS=['Reducida · una','Reducida · tres','Amplia · una','Amplia · tres']
MODELS={'unet':'U-Net','segmenter':'Segmenter'}


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def csv_file(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def markdown_table(rows,columns):
    lines=['| '+' | '.join(title for _,title in columns)+' |',
           '| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        lines.append('| '+' | '.join(f'{row[key]:.2f}' if isinstance(row[key],float) else str(row[key]) for key,_ in columns)+' |')
    return '\n'.join(lines)


def effects(primary):
    """Diferencias en puntos porcentuales, sin inferencia estadística."""
    get=lambda c,m:next(r for r in primary if r['condicion']==c and r['arquitectura']==m)
    result=[]
    metrics=['miou','f1_macro','f1_Mani','precision_mani','recall_mani']
    contrasts=[('Campañas: reducida','reducida_tres','reducida_2020_2021'),
               ('Campañas: amplia','amplia_tres','amplia_2020_2021'),
               ('Cobertura: una campaña','amplia_2020_2021','reducida_2020_2021'),
               ('Cobertura: tres campañas','amplia_tres','reducida_tres')]
    for m in MODELS:
        for name,plus,minus in contrasts:
            result.append({'arquitectura':m,'contraste':name,
                           **{k:get(plus,m)[k]-get(minus,m)[k] for k in metrics}})
        result.append({'arquitectura':m,'contraste':'Interacción: efecto campañas amplia menos reducida',
            **{k:(get('amplia_tres',m)[k]-get('amplia_2020_2021',m)[k])-(get('reducida_tres',m)[k]-get('reducida_2020_2021',m)[k]) for k in metrics}})
    for c in CONDITIONS:
        result.append({'arquitectura':'Segmenter menos U-Net','contraste':c,
                       **{k:get(c,'segmenter')[k]-get(c,'unet')[k] for k in metrics}})
    return result


def confusion(cm,title,path):
    cm=np.asarray(cm,dtype=np.int64)
    # NoData en referencia se ignora; predicción 0 válida conserva su error.
    cols=list(range(5)) if cm[1:,0].any() else list(range(1,5))
    values=cm[1:,cols]
    norm=100*values/np.maximum(cm[1:].sum(1,keepdims=True),1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,v,label in zip(axes,[values,norm],['Conteos de píxeles','Porcentaje por referencia']):
        im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=100)
        for i in range(4):
            for j in range(len(cols)):
                text=f'{v[i,j]:,.0f}' if v is values else f'{v[i,j]:.2f}%'
                ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if norm[i,j]>55 else 'black')
        ax.set_xticks(range(len(cols)),[NAMES[i] for i in cols]);ax.set_yticks(range(4),NAMES[1:])
        ax.set_xlabel('Predicción');ax.set_ylabel('Referencia MNC');ax.set_title(label)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def render_comparison(base,out,condition,year,tile,maskbase,window=None):
    size=(768,768)
    with rasterio.open(files(year,tile)[2]) as src:
        rgb=src.read([3,2,1],window=window,out_shape=(3,*size)).astype(float).transpose(1,2,0)
    for c in range(3):
        v=rgb[:,:,c][rgb[:,:,c]>0];lo,hi=np.percentile(v,[2,98]) if v.size else (0,1)
        rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1),0,1)
    with rasterio.open(maskbase/'mascaras'/f'{year}_{tile}_independiente.tif') as src:
        target=src.read(1,window=window,out_shape=size,resampling=Resampling.nearest)
    predictions={}
    for model in MODELS:
        pred=base/'experimentos'/f'{condition}_{model}'/'evaluacion_test_actual/modelos/principal/predicciones'/year/f'prediccion_{tile}.tif'
        with rasterio.open(pred) as src:
            predictions[model]=src.read(1,window=window,out_shape=size,resampling=Resampling.nearest)
    def color(labels):
        a=COLORS[labels].copy();a[target==0]=[65,65,65];return a
    def error(labels):
        a=np.full((*size,3),65,dtype=np.uint8)
        a[(target>0)&(labels==target)]=[0,170,0];a[(target>0)&(labels!=target)]=[220,0,0];return a
    panels=[('RGB febrero',rgb),('Referencia independiente',color(target)),
            ('U-Net',color(predictions['unet'])),('Segmenter',color(predictions['segmenter'])),
            ('U-Net: verde acierto / rojo error',error(predictions['unet'])),
            ('Segmenter: verde acierto / rojo error',error(predictions['segmenter']))]
    mode='detalle_mani' if window is not None else 'panorama'
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    for ax,(name,a) in zip(axes.flat,panels):ax.imshow(a);ax.set_title(name);ax.axis('off')
    fig.suptitle(f'{condition} · {year.replace("_","/")} · {tile} · {mode}\nGris: excluido/NoData de referencia')
    fig.legend(handles=[Patch(color=COLORS[i]/255,label=NAMES[i]) for i in range(1,5)],loc='lower center',ncol=4)
    fig.tight_layout(rect=(0,.04,1,.95))
    path=out/'figuras'/f'{mode}_{condition}_{year}_{tile}.png'
    fig.savefig(path,dpi=130);plt.close(fig)
    return str(path.relative_to(out))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
    a=p.parse_args(argv);base=a.base;out=a.salida
    state=json.loads((base/'estado.json').read_text());assert state['estado']=='finalizado' and state['modelos']==8
    assert not out.exists(),'Conservar reportes previos; elegir salida nueva'
    protocol=json.loads((base/'protocolo.json').read_text())
    data=json.loads((base/'comparacion_ocho_modelos.json').read_text())
    assert len(data)==64 and len({r['experimento'] for r in data})==8
    sources={};meta=[];rows=[];curves={};maskbase=None
    configs={}
    for job in protocol['jobs']:
        exp=base/'experimentos'/job['nombre'];result=json.loads((exp/'resultado_supervisor.json').read_text())
        assert result=={'entrenamiento':0,'evaluacion':0}
        cfg=json.loads((exp/'config_multicampana.json').read_text());configs[job['nombre']]=cfg
        assert cfg['seleccion_train_sha256']==job['sha256'] and cfg['epochs']==100 and cfg['batch_size']==32 and cfg['seed']==42
        h=np.load(exp/'historial.npy');assert h.shape==(100,5) and np.isfinite(h).all()
        best=h[h[:,3].argmin()];curves[job['nombre']]=h
        val=json.loads((exp/'metricas_multicampana.json').read_text())['val']['global']
        meta.append({'experimento':job['nombre'],'epocas':len(h),'epoca_minima_val':int(best[0]),'perdida_val_minima':float(best[3]),'accuracy_val':val['accuracy']*100,'miou_val':val['miou']*100,'f1_mani_val':val['clases']['Mani']['f1']*100})
        ev=json.loads((exp/'evaluacion_test_actual/evaluacion_test_actual.json').read_text())
        newbase=Path(ev['protocolo']['preparacion_base'])
        assert maskbase is None or maskbase==newbase;maskbase=newbase
        sources[job['nombre']]={'config':cfg,'checkpoint_sha256':ev['protocolo']['modelos']['principal']['sha256']}
    for condition in CONDITIONS:
        u=configs[condition+'_unet'];s=configs[condition+'_segmenter']
        assert all(u[k]==s[k] for k in u if k!='modelo'),'Configuraciones no pareadas'
    for g in data:
        reference=next(r for r in data if r['campana']==g['campana'] and r['criterio']==g['criterio'])
        assert g['pixeles']==reference['pixeles']
        # La referencia y soporte por clase también deben coincidir.
        assert np.array_equal(np.array(g['confusion']).sum(1),np.array(reference['confusion']).sum(1))
        rows.append({**{k:g[k] for k in ['experimento','arquitectura','condicion','campana','criterio','pixeles']},
                     **{k:100*g[k] for k in ['accuracy','miou','f1_macro']},
                     **{'f1_'+k:100*v['f1'] for k,v in g['clases'].items()},
                     'precision_mani':100*g['clases']['Mani']['precision'],
                     'recall_mani':100*g['clases']['Mani']['recall'],
                     'lotes_referencia':'no disponibles; métricas por píxel'})
    primary=[r for r in rows if r['campana']=='2021_2022' and r['criterio']=='independiente']
    primary.sort(key=lambda r:(CONDITIONS.index(r['condicion']),r['arquitectura']!='unet'))
    contrasts=effects(primary)
    out.mkdir(parents=True);(out/'figuras').mkdir();(out/'metricas').mkdir()
    save(out/'metricas/resultados_completos.json',data);save(out/'metricas/procedencia_checkpoints.json',sources)
    save(out/'metricas/resumen_2021_2022.json',primary)
    csv_file(out/'metricas/todas_las_metricas.csv',rows);csv_file(out/'metricas/resumen_2021_2022.csv',primary)
    csv_file(out/'metricas/contrastes_2021_2022_pp.csv',contrasts);csv_file(out/'metricas/validacion_y_epocas.csv',meta)
    for n in ['protocolo.json','auditoria_mani.json','mapa_nucleo_mani.png']:
        shutil.copy2(base/n,out/n)
    groups={'Resumen':[],'Curvas VAL':[],'Matrices de confusión':[],'Panoramas':[],'Detalles de maní':[]}
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,key,title in zip(axes,['miou','f1_macro','f1_Mani'],['mIoU (%)','F1 macro (%)','F1 maní (%)']):
        for i,(model,label) in enumerate(MODELS.items()):
            values=[next(r[key] for r in primary if r['condicion']==c and r['arquitectura']==model) for c in CONDITIONS]
            bars=ax.bar(np.arange(4)+(i-.5)*.36,values,.36,label=label)
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
        ax.set_xticks(np.arange(4),LABELS,rotation=20,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.margins(y=.18)
    fig.suptitle('2021/2022 no vista · TEST independiente · 3072 parches por modelo')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(out/'figuras/resumen_2021_2022.png',dpi=150);plt.close(fig)
    groups['Resumen'].append('figuras/resumen_2021_2022.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,key,title in zip(axes,['miou','f1_Mani'],['mIoU (%)','F1 maní (%)']):
        for model,label in MODELS.items():
            for domain,style in [('reducida','-'),('amplia','--')]:
                pair=[next(r[key] for r in primary if r['arquitectura']==model and r['condicion']==domain+'_'+t) for t in ['2020_2021','tres']]
                ax.plot([1,3],pair,style,marker='o',label=label+' · '+domain)
        ax.set_xticks([1,3],['Una campaña','Tres campañas']);ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Respuesta a campañas según arquitectura y cobertura · diferencias descriptivas')
    fig.tight_layout();fig.savefig(out/'figuras/interacciones.png',dpi=150);plt.close(fig)
    groups['Resumen'].append('figuras/interacciones.png')
    years=protocol['campanas'];years=[y for y,_ in years]+['2021_2022']
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for condition in CONDITIONS:
        for model,label in MODELS.items():
            rr=[next(r for r in rows if r['condicion']==condition and r['arquitectura']==model and r['campana']==y and r['criterio']=='independiente') for y in years]
            for ax,key in zip(axes,['miou','f1_Mani']):ax.plot(range(4),[r[key] for r in rr],marker='o',label=condition+' '+label)
    for ax,title in zip(axes,['mIoU (%)','F1 maní (%)']):ax.set_xticks(range(4),[y.replace('_','/') for y in years],rotation=20);ax.set_title(title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'figuras/campanas_test.png',dpi=150);plt.close(fig)
    groups['Resumen'].append('figuras/campanas_test.png')
    for condition in CONDITIONS:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for model,label in MODELS.items():
            h=curves[condition+'_'+model]
            axes[0].plot(h[:,0],h[:,3],label=label);axes[1].plot(h[:,0],100*h[:,4],label=label)
            best=int(h[:,3].argmin());axes[0].scatter(h[best,0],h[best,3],s=25)
        for ax,title in zip(axes,['Pérdida VAL · punto: mínimo','Accuracy VAL (%)']):ax.set_title(title);ax.set_xlabel('Época');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(condition);fig.tight_layout();name=f'figuras/curvas_{condition}.png';fig.savefig(out/name,dpi=150);plt.close(fig);groups['Curvas VAL'].append(name)
    for g in data:
        if g['criterio']!='independiente':continue
        name=f'figuras/confusion_{g["experimento"]}_{g["campana"]}.png'
        confusion(g['confusion'],g['experimento']+' · '+g['campana']+' · independiente',out/name)
        groups['Matrices de confusión'].append(name)
    crop_records=[]
    for year in years:
        for tile in ['20HNJ','20HNH']:
            crop=None
            if year=='2021_2022':
                with rasterio.open(maskbase/'mascaras'/f'{year}_{tile}_independiente.tif') as src:
                    best=max(((int(np.count_nonzero(src.read(1,window=w)==4)),w) for _,w in src.block_windows(1)),key=lambda v:v[0])
                    w=best[1];x=int(w.col_off+w.width/2);y=int(w.row_off+w.height/2)
                    crop=Window(max(0,min(src.width-1024,x-512)),max(0,min(src.height-1024,y-512)),min(1024,src.width),min(1024,src.height))
                crop_records.append({'campana':year,'tile':tile,'window':list(crop.flatten()),'regla':'Bloque con más maní de referencia; sin consultar modelos','max_mani_px_bloque':best[0]})
            for condition in CONDITIONS:
                name=render_comparison(base,out,condition,year,tile,maskbase)
                groups['Panoramas'].append(name);print(name,flush=True)
                if crop is not None:
                    name=render_comparison(base,out,condition,year,tile,maskbase,crop)
                    groups['Detalles de maní'].append(name);print(name,flush=True)
    save(out/'metricas/ventanas_detalle.json',crop_records)
    columns=[('condicion','Condición'),('arquitectura','Modelo'),('accuracy','Accuracy'),('miou','mIoU'),('f1_macro','F1 macro'),('f1_Mani','F1 maní'),('precision_mani','Precisión maní'),('recall_mani','Recall maní')]
    text=['# Ocho modelos: U-Net y Segmenter, cobertura y campañas — 22/10/2026','',
        'Ocho entrenamientos de 100 épocas y ocho evaluaciones terminados con código 0. Duración total: 14 h 40 min 23 s. Fin: 21/10 a las 23:50 de Argentina. No se reentrenó para este reporte.','',
        '## Comparación principal: 2021/2022 no vista','',
        'TEST 20HNJ/20HNH, exclusión común TRAIN/VAL, 177.822.530 píxeles válidos. Todas las métricas están en porcentaje. Checkpoints seleccionados por VAL.', '',markdown_table(primary,columns),'',
        '## Lectura de los resultados','',
        '- La mayor mIoU observada corresponde a U-Net amplia con tres campañas (54,57%). Segmenter supera a U-Net con una campaña en ambas coberturas, pero U-Net lo supera con tres. No hay una ventaja uniforme de arquitectura.',
        '- Agregar campañas mejora mIoU de U-Net en 2,39 puntos con cobertura reducida y 4,81 con amplia. En Segmenter cambia +0,26 y −1,34 puntos respectivamente. Ampliar cobertura no ayuda de manera uniforme.',
        '- Maní no está resuelto: el mayor F1 ronda 10,74% (Segmenter reducida/una y U-Net reducida/tres son casi iguales). Las precisiones de todos los modelos son menores al 6%. El mayor recall, 86,94%, corresponde a Segmenter reducida/tres, pero con precisión 4,73%. Detectar más maní no implica un mapa más confiable.',
        '- Para desempeño general, conservaría U-Net amplia/tres como referencia observada; antes de otro entrenamiento revisaría etiquetas, falsos positivos y desempeño por lote de maní. Las diferencias pequeñas no justifican elegir un ganador definitivo.', '',
        '## Contrastes en puntos porcentuales','',markdown_table(contrasts,[('arquitectura','Arquitectura'),('contraste','Contraste'),('miou','Δ mIoU'),('f1_Mani','Δ F1 maní')]),'',
        '## Controles y límites','',
        'Se verificaron ocho historiales de 100 épocas, códigos 0 de entrenamiento y evaluación, coincidencia de configuraciones por pares salvo arquitectura, firmas de selección y el mismo soporte de referencia. Cada modelo usa 3072 parches; la mezcla asigna 1024 por campaña y conserva ubicaciones dentro de cada cobertura. Pesos comunes: '+str(protocol['pesos_comunes'])+'.', '',
        'Fondo incluye otros cultivos y coberturas; no significa exclusivamente no agrícola. NoData de referencia se ignora. Las matrices muestran cuatro clases válidas y conservan una columna NoData predicha cuando hay errores de ese tipo; no se eliminan para mejorar métricas.', '',
        'Las cifras principales suman las matrices de ambos tiles (no promedian sus F1). Los CSV distinguen mapa completo e independiente. La referencia es MNC, no verificación de campo independiente. En TRAIN se ignoraron positivos IPAVE que discrepan con MNC; VAL/TEST permanecieron originales. Cambiaron también los pesos comunes frente al piloto anterior: sin un control adicional no se puede atribuir una diferencia entre ensayos solo a IPAVE.', '',
        'Una semilla por celda: sin pruebas de significación ni intervalos entre semillas. La cobertura describe un dominio geográfico concreto, no área total observada adicional. Se igualó el número de parches, no píxeles de maní ni número de píxeles válidos después de ignorar conflictos. La misma semilla no garantiza la misma secuencia de aumentos entre arquitecturas. Igual número de épocas no asegura igual convergencia; no hubo búsqueda de hiperparámetros por arquitectura. TEST se consultó en trabajos previos, por lo que el estudio es exploratorio.', '',
        'Las panorámicas están reducidas para visualizar; detalles de maní elegidos por referencia y comunes a todos los modelos, no muestras aleatorias. Gris: excluido/NoData; verde: acierto; rojo: error. RGB de febrero. No se transfieren pesos ni TIFF al mirror.', '',
        '## Archivos','',
        '- `index.html`: galería local con filtros por condición, campaña y tile.',
        '- `metricas/todas_las_metricas.csv`: 64 filas, los dos criterios y todas las campañas.',
        '- `metricas/contrastes_2021_2022_pp.csv`: efectos descriptivos de cobertura, campañas, arquitectura e interacción.',
        '- `metricas/validacion_y_epocas.csv`: épocas de menor pérdida y métricas VAL.',
        '- `metricas/resultados_completos.json`: matrices completas y métricas originales.',
        '- `metricas/procedencia_checkpoints.json`: configuraciones y firmas de pesos.',
        '- `sha256.json`: comprobación de integridad de esta copia.', '',
        'Fuente física: `'+str(base)+'`. Figuras y métricas generadas desde resultados congelados; no se eligieron otros pesos con TEST.']
    report='\n'.join(text)+'\n';(out/'RESULTADOS.md').write_text(report)
    table='<table><thead><tr>'+''.join('<th>'+html.escape(t)+'</th>' for _,t in columns)+'</tr></thead><tbody>'
    for r in primary:table+='<tr>'+''.join('<td>'+html.escape(f'{r[k]:.2f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k,_ in columns)+'</tr>'
    table+='</tbody></table>'
    gallery=[]
    for group,names in groups.items():
        items=''.join('<article data-name="'+html.escape(name)+'"><h3>'+html.escape(Path(name).stem)+'</h3><a href="'+name+'"><img loading="lazy" src="'+name+'"></a></article>' for name in names)
        gallery.append('<details '+('open' if group=='Resumen' else '')+'><summary>'+html.escape(group)+' ('+str(len(names))+')</summary>'+items+'</details>')
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Factorial U-Net y Segmenter</title><style>body{font:16px system-ui;margin:24px auto;padding:0 16px;max-width:1550px;color:#182739}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd}th{text-align:left;background:#eaf0f7}img{max-width:100%}summary{font-size:22px;cursor:pointer;padding:16px;background:#edf3f8;margin-top:20px}input{padding:12px;width:90%;margin:20px 0}article{margin:26px 0}h3{font-size:15px} .scroll{overflow-x:auto}</style><h1>U-Net y Segmenter: cobertura × campañas</h1><p>Ocho modelos · 3072 parches · 100 épocas · TEST independiente · referencia MNC.</p><p><strong>Resultado general:</strong> U-Net amplia/tres alcanza la mayor mIoU (54,57%). Segmenter supera a U-Net con una campaña; U-Net lo supera con tres. Maní sigue con precisión inferior al 6% en todos los modelos. Una sola semilla: diferencias exploratorias.</p><p><a href="RESULTADOS.md">Informe completo</a> · <a href="metricas/todas_las_metricas.csv">Métricas CSV</a> · <a href="metricas/contrastes_2021_2022_pp.csv">Contrastes CSV</a></p><h2>2021/2022 no vista · porcentajes</h2><div class="scroll">'''+table+'''</div><label>Filtrar figuras por texto (ej.: amplia_tres, 2021_2022, 20HNH):<input id="filter" type="search"></label>'''+''.join(gallery)+'''<script>document.getElementById('filter').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.name.toLowerCase().includes(q));if(q)document.querySelectorAll('details').forEach(d=>d.open=true)})</script></html>'''
    (out/'index.html').write_text(page)
    shutil.copy2(Path(__file__),out/'generador_reporte.py')
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    save(out/'sha256.json',manifest)
    print('REPORTE',out,'PNG',len(list((out/'figuras').glob('*.png'))),'archivos',len(manifest)+1,flush=True)


if __name__=='__main__':main()
