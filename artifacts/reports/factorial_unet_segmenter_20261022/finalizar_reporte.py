from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path.home()/'cordoba_ia_unne'
out=Path('/mnt/yacy_3/prod_bis/ferreyra/dataset/factorial_unet_segmenter_mani_20261021_v1/resultados_20261022_v1')
assert (out/'sha256.json').exists()
source=root/'scripts/diagnostico/reportar_factorial_arquitecturas.py'
s=source.read_text().replace("['Efecto en mIoU (puntos)','Efecto en F1 maní (puntos)']","['mIoU (%)','F1 maní (%)']")
source.write_text(s)
shutil.copy2(source,out/'generador_reporte.py')
rows=json.loads((out/'metricas/resumen_2021_2022.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,key,title in zip(axes,['miou','f1_Mani'],['mIoU (%)','F1 maní (%)']):
    for model,label in [('unet','U-Net'),('segmenter','Segmenter')]:
        for domain,style in [('reducida','-'),('amplia','--')]:
            pair=[next(r[key] for r in rows if r['arquitectura']==model and r['condicion']==domain+'_'+t) for t in ['2020_2021','tres']]
            ax.plot([1,3],pair,style,marker='o',label=label+' · '+domain)
    ax.set_xticks([1,3],['Una campaña','Tres campañas']);ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Respuesta a campañas según arquitectura y cobertura · diferencias descriptivas')
fig.tight_layout();fig.savefig(out/'figuras/interacciones.png',dpi=150);plt.close(fig)
shutil.copy2(root/'docs/historico/factorial_unet_segmenter_mani_20261021.md',out/'PROTOCOLO_Y_RESULTADOS.md')
note='Los mínimos de pérdida VAL de Segmenter aparecen en las épocas 90–99; los de U-Net en 16–57. No se demostró convergencia equivalente. El F1 de maní de VAL fue 44,45–57,18%, frente a 1,36–10,74% en este TEST: hay una brecha importante que requiere revisión por lote.'
p=out/'RESULTADOS.md';s=p.read_text()
if note not in s:p.write_text(s+'\n## Nota adicional de validación\n\n'+note+'\n')
p=out/'index.html';s=p.read_text()
if note not in s:p.write_text(s.replace('<h2>2021/2022', '<p>'+note+'</p><p><a href="PROTOCOLO_Y_RESULTADOS.md">Protocolo y análisis ampliado</a></p><h2>2021/2022',1))

conditions=["reducida_2020_2021","reducida_tres","amplia_2020_2021","amplia_tres"]
labels=["Reducida · una","Reducida · tres","Amplia · una","Amplia · tres"]
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
for ax,key,title in zip(axes,["miou","f1_macro","f1_Mani"],["mIoU (%)","F1 macro (%)","F1 maní (%)"]):
 for i,(model,label) in enumerate([("unet","U-Net"),("segmenter","Segmenter")]):
  values=[next(r[key] for r in rows if r["condicion"]==c and r["arquitectura"]==model) for c in conditions]
  bars=ax.bar(np.arange(4)+(i-.5)*.36,values,.36,label=label)
  ax.bar_label(bars,fmt="%.2f",fontsize=8,padding=2)
 ax.set_xticks(np.arange(4),labels,rotation=20,ha="right");ax.set_title(title);ax.grid(axis="y",alpha=.2);ax.margins(y=.18)
fig.suptitle("2021/2022 no vista · TEST independiente · 3072 parches por modelo")
fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.94),ncol=2)
fig.tight_layout(rect=(0,0,1,.88));fig.savefig(out/"figuras/resumen_2021_2022.png",dpi=150);plt.close(fig)

manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='sha256.json'}
(out/'sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('FINALIZADO',len(manifest)+1,'archivos',len(list((out/'figuras').glob('*.png'))),'figuras')
