"""Matrices y metricas; filas=referencia, columnas=prediccion.

Referencia 0 se ignora; prediccion 0 sobre referencia valida es error.
Mantener separados codigos crudos MNC y clases ya remapeadas del TEST.
"""
import numpy as np
import rasterio
from utils.leyendas_mnc import verificar_codigos


def metricas_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm[1:, :].sum())
    resultado = {'pixeles': total, 'accuracy': float(np.trace(cm[1:, 1:]) / total) if total else 0.0,
                 'confusion': cm.tolist(), 'clases': {}}
    f1s, ious = [], []
    for c, nombre in enumerate(['Fondo', 'Maiz', 'Soja', 'Mani'], 1):
        tp = int(cm[c, c]); reales = int(cm[c, :].sum()); predichos = int(cm[1:, c].sum())
        precision = tp / predichos if predichos else 0.0
        recall = tp / reales if reales else 0.0
        f1 = 2 * tp / (reales + predichos) if reales + predichos else 0.0
        iou = tp / (reales + predichos - tp) if reales + predichos - tp else 0.0
        resultado['clases'][nombre] = {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
                                      'pixeles': reales}
        if reales:
            f1s.append(f1); ious.append(iou)
    resultado['f1_macro'] = float(np.mean(f1s)) if f1s else 0.0
    resultado['miou'] = float(np.mean(ious)) if ious else 0.0
    return resultado


def confusion_mnc(pred_path,mask_path,remap,year):
    cm = np.zeros((5,5),dtype=np.int64)
    with rasterio.open(pred_path) as pred, rasterio.open(mask_path) as mask:
        if (pred.shape,pred.transform,pred.crs)!=(mask.shape,mask.transform,mask.crs):
            raise ValueError('Grilla distinta: '+str(pred_path))
        for _,window in pred.block_windows(1):
            raw = mask.read(1,window=window)
            if year=='2020_2021': verificar_codigos(np.unique(raw),ignorar_codigo20=True)
            y = remap[raw]
            p = pred.read(1,window=window)
            if p.max()>4: raise ValueError('Prediccion fuera de clases 0..4')
            valid = y>0  # misma mascara MNC para todos; pred=0 cuenta como error
            cm += np.bincount(y[valid].astype(np.int64)*5+p[valid],minlength=25).reshape(5,5)
    return cm


def confusion_remapeada(pred,mask):
 cm=np.zeros((5,5),dtype=np.int64)
 with rasterio.open(pred) as p,rasterio.open(mask) as m:
  if (p.crs,p.transform,p.shape)!=(m.crs,m.transform,m.shape):raise ValueError('Grillas diferentes')
  for _,win in p.block_windows(1):
   a=p.read(1,window=win);b=m.read(1,window=win)
   if a.max()>4 or b.max()>4:raise ValueError('Clases fuera de 0..4')
   valid=b>0;cm+=np.bincount(b[valid].astype(np.int64)*5+a[valid],minlength=25).reshape(5,5)
 return cm
