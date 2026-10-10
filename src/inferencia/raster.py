"""Inferencia por bloques, sin seleccionar modelos ni protocolos."""
from contextlib import ExitStack
import time
import numpy as np
import rasterio
from rasterio.windows import Window
import torch


def normalizar(array):
    # Igual al dataset: dividir /10000, almacenar f16 y convertir a f32.
    x = (array.astype(np.float32)/10000).astype(np.float16).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Composite con NaN/Inf o valores fuera de float16')
    return x


def inferir_tile(model, archivos, destino, device, batch=4, size=512, tags=None):
    destino.parent.mkdir(parents=True,exist_ok=True)
    partial = destino.with_suffix('.partial.tif')
    tags = tags or {}
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(path)) for path in archivos]
            ref = sources[0]
            grid = (ref.shape,ref.transform,ref.crs)
            if any((src.shape,src.transform,src.crs)!=grid or src.count!=4 for src in sources):
                raise ValueError('Composites incompatibles en grilla/bandas')
            profile = ref.profile.copy()
            profile.update(count=1,dtype='uint8',nodata=0,compress='deflate',predictor=1,
                           tiled=True,blockxsize=256,blockysize=256)
            windows = [Window(x,y,min(size,ref.width-x),min(size,ref.height-y))
                       for y in range(0,ref.height,size) for x in range(0,ref.width,size)]
            with rasterio.open(partial,'w',**profile) as dst, torch.inference_mode():
                dst.update_tags(**tags)
                for offset in range(0,len(windows),batch):
                    selected = windows[offset:offset+batch]
                    inputs, missing = [],[]
                    for window in selected:
                        x = normalizar(np.concatenate([src.read(window=window) for src in sources],axis=0))
                        missing.append((x==0).all(axis=0))
                        h,w = x.shape[-2:]
                        inputs.append(np.pad(x,((0,0),(0,size-h),(0,size-w)),mode='edge'))
                    tensor = torch.from_numpy(np.stack(inputs)).to(device)
                    with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                        logits = model(tensor)
                    if not torch.isfinite(logits).all():
                        raise ValueError('El modelo produjo NaN/Inf')
                    pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    for i,window in enumerate(selected):
                        result = pred[i,:int(window.height),:int(window.width)].copy()
                        result[missing[i]]=0
                        dst.write(result,1,window=window)
                    if offset==0 or (offset//batch)%20==0 or offset+batch>=len(windows):
                        print(f'  {destino.name}: {min(offset+batch,len(windows))}/{len(windows)} bloques, '
                              f'{time.monotonic()-started:.0f}s',flush=True)
        partial.replace(destino)
    except BaseException:
        if partial.exists(): partial.unlink()
        raise
    return time.monotonic()-started
