"""
Entrenamiento de clasificacion de cultivos de verano sobre Cordoba completa.
Dataset: composites Sentinel-2 2019-2020 + etiquetas MNC INTA verano 2020,
esquema de 5 clases (sin Sorgo -- ver src/README_MNC.md: es una clase de una
sola zona y el propio informe INTA reporta baja exactitud ahi).

Los datos ya tienen el remapeo a 5 clases aplicado, por eso NO se vuelve a
remapear al cargar. Requiere haber convertido los .npz de origen a pares
<tile>_X.npy/_Y.npy sueltos (ver src/03b_convertir_datasets_a_mmap.py), que
es donde se valida una sola vez el flag remap_aplicado=1 de los .npz.

Es el pipeline principal del repo (reemplaza a las variantes chicas
1_train_mnc2.py / 1_train_multitile.py, y a la version separada con
augmentacion 1_train_cordoba_aug.py -- ahora todo vive en un unico script).

Uso:
    python3 src/03b_convertir_datasets_a_mmap.py /mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16
    python3 src/1_train_cordoba.py [--modelo unet|transunet]

--modelo transunet usa utils/model.py::TransUNet (SimpleUNet + self-attention
global en el bottleneck, ver docstring de la clase) en vez del baseline, y
escribe en exp_cordoba_transunet_f16 para no pisar el checkpoint del baseline.

--modelo utae usa utils/model.py::UTAE (U-Net temporal, ver docstring de la
clase): en vez de aplanar los 7 meses como canales sueltos, corre el encoder
por mes (T plegado en el batch) y colapsa el tiempo con L-TAE. El encoder T
veces por batch pesa mucho mas en memoria -- BATCH_SIZE se reduce a 16 (ver
smoke test que sondeo el batch maximo en la RTX 3090) solo para este modelo.
Escribe en exp_cordoba_utae_f16.

--modelo fcrtransunet usa utils/model.py::FCRTransUNet (adaptacion de Han et
al. 2026, MDPI Agriculture 16(7):727, ver docstring de la clase): TransUNet
+ FEM (self-attention no-local + gated conv) en el nivel 3 del encoder +
Coordinate Attention en los 4 skips + Reverse Attention en la segunda etapa
del decoder. Escribe en exp_cordoba_fcrtransunet_f16.

--dir-train / --dir-exp / --sin-2017 permiten entrenar sobre otro dataset sin
tocar los defaults. Ventana dic-may, para inferir sobre la campana 2018-2019
(descargada solo dic 2018 - may 2019):
    python3 src/1_train_cordoba.py --modelo unet --sin-2017 \
        --dir-train /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_f16 \
        --dir-exp   /mnt/yacy_3/prod_bis/ferreyra/dataset/exp_cordoba_dicmay_unet_f16
(sin la campana 2017-2018: sus composites no tienen may-2018).

--dir-train-extra suma a TRAIN los mismos tiles de TRAIN de otro dataset (otra
campana, misma ventana y canales); val/test no cambian. --pesos reemplaza los
pesos por clase (calcularlos con 0b_calcular_pesos.py --dirs ...). Ventana
dic-may con la campana 2017-2018 (Nivel3) sumada, ver pipeline_dicmay_1718.sh:
    python3 src/1_train_cordoba.py --modelo unet --sin-2017 \
        --dir-train       /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_f16 \
        --dir-train-extra /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_1718_f16 \
        --pesos 0 0.8 1.0 0.8 4.2 \
        --dir-exp /mnt/yacy_3/prod_bis/ferreyra/dataset/exp_cordoba_dicmay_1718_unet_f16
"""
import argparse
import json
from pathlib import Path
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet, TransUNet, UTAE, FCRTransUNet
from utils.tile_dataset import TileDatasetMmap
from utils.losses import CEDiceLoss
from utils.multicampana import (SPLIT, normalizar_campanas, validar_campanas,
                                cargar_campanas, pesos_desde_conteos, metricas_confusion)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--modelo", choices=["unet", "transunet", "utae", "fcrtransunet"],
                     default="unet",
                     help="unet=SimpleUNet (baseline), transunet=SimpleUNet con "
                          "self-attention global en el bottleneck, utae=U-Net "
                          "temporal con L-TAE, fcrtransunet=TransUNet + FEM/CA/RAM "
                          "(ver utils/model.py)")
parser.add_argument("--dir-train", default=None,
                     help="dir con los dataset_<TILE>_X/_Y.npy (default: train_cordoba_f16)")
parser.add_argument("--dir-exp", default=None,
                     help="dir de salida de pesos/historial (default: exp_cordoba_[<modelo>_]f16)")
parser.add_argument("--sin-2017", action="store_true",
                     help="no sumar los tiles de la campana 2017-2018 a TRAIN")
parser.add_argument("--dir-train-extra", default=None,
                     help="dir con dataset_<TILE>_X/_Y.npy de otra campana: sus tiles de "
                          "TRAIN se suman a TRAIN (val/test siguen saliendo de --dir-train)")
parser.add_argument("--pesos", nargs=5, type=float, default=None,
                     help="pesos por clase [NoData Fondo Maiz Soja Mani] (default: los de abajo)")
parser.add_argument("--campana", nargs=2, action="append", metavar=("AAAA_AAAA", "DIRECTORIO"),
                    help="repetible: incorpora TRAIN/VAL/TEST de cada campana")
parser.add_argument("--verificar-datos", action="store_true",
                    help="valida multicampana y pesos, sin crear salidas ni entrenar")
ARGS = parser.parse_args()
if ARGS.campana:
    if ARGS.dir_train or ARGS.dir_train_extra or ARGS.sin_2017:
        parser.error("--campana no se combina con --dir-train/--dir-train-extra/--sin-2017")
    try:
        ARGS.campana = normalizar_campanas(ARGS.campana)
    except ValueError as exc:
        parser.error(str(exc))
elif ARGS.verificar_datos:
    parser.error("--verificar-datos requiere --campana")
if ARGS.pesos is not None and (not np.isfinite(ARGS.pesos).all()
                              or min(ARGS.pesos) < 0 or ARGS.pesos[0] != 0
                              or max(ARGS.pesos[1:]) <= 0):
    parser.error("--pesos requiere cinco valores finitos no negativos; NoData=0")

BASE_DIR   = "/mnt/yacy_1/prod/ferreyra/dataset"
DIR_TRAIN  = ARGS.dir_train or f"{BASE_DIR}/train_cordoba_f16"
# exp_cordoba_f16 se deja intacto (compatibilidad con 2_inferencia.py y demas
# scripts que ya apuntan ahi) -- las variantes escriben en dirs aparte para no
# pisar el checkpoint del baseline.
DIR_EXP    = f"{BASE_DIR}/exp_cordoba_f16" if ARGS.modelo == "unet" \
             else f"{BASE_DIR}/exp_cordoba_{ARGS.modelo}_f16"
if ARGS.campana:
    DIR_EXP = str(Path(ARGS.campana[0][1]).parent /
                  ("exp_multicampana_" + "_".join(n for n, _ in ARGS.campana) + "_" + ARGS.modelo))
DIR_EXP    = ARGS.dir_exp or DIR_EXP
PREFIJO    = "dataset_"

# Split por franjas de cobertura de cultivo (ver analisis de cobertura MNC)
# Split espacial por tiles completos, balanceado en tres criterios:
#   1) gradiente geografico oeste-centro-este (evita mismatch de dominio)
#   2) presencia de Mani (clase minoritaria) en los tres conjuntos
#   3) validacion ampliada a 4 tiles para reducir el ruido de la curva Val
# Split de 17 tiles (11 originales + 6 sumados via
# 03c_generar_dataset_cordoba_f16.py, verificados con 0_verificar_pipeline.py).
# Los 6 nuevos se suman todos a TRAIN -- val/test quedan igual que antes para
# poder comparar resultados contra las corridas previas con el mismo test set.
TILES_TRAIN = list(SPLIT['train'])
TILE_VAL = list(SPLIT['val'])
TILES_TEST = list(SPLIT['test'])

# Campana 2017-2018 (legacy, ver 03d_generar_dataset_cordoba_2017_f16.py):
# mismos tiles Sentinel-2 que algunos de arriba pero de otra campana agricola
# (etiquetas Nivel3 remapeadas a las 5 clases MNC vigentes), con sufijo
# "_2017" en el nombre de archivo para no pisar el dataset 2019-2020 de esos
# mismos tiles. Meses alineados a la misma ventana del ciclo agricola que la
# campana 2020 (oct-abr, ver 03d) -- de los 6 tiles con mascara, solo estos 4
# tienen esa ventana completa (20JML y 20HNK quedan afuera, les faltan meses).
# Se suman todos a TRAIN (mas diversidad interanual) -- val/test quedan
# intactos, igual que con los 6 tiles de 03c.
TILES_TRAIN_2017 = ['20HLJ_2017', '20HLK_2017', '20JLL_2017', '20JNL_2017']
if not ARGS.sin_2017 and not ARGS.campana:
    TILES_TRAIN = TILES_TRAIN + TILES_TRAIN_2017

NUM_CLASSES   = 5
# utae corre el encoder T=7 veces por batch (una por mes) -- a batch 32 la
# RTX 3090 tira OOM; 16 usa ~17GB de pico, medido con smoke test.
BATCH_SIZE    = 16 if ARGS.modelo == "utae" else 32
EPOCHS        = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-3
PATIENCE      = 30
SEED          = 42

# floryacy: 32 nucleos de CPU / RTX 3090. Con TileDatasetMmap cada __getitem__
# lee del disco bajo demanda (antes el dataset ya estaba entero en RAM), asi
# que varios workers en paralelo si aportan (se solapan con el computo en GPU).
NUM_WORKERS = 16
PREFETCH_FACTOR = 4

# Pesos por clase — recalculados con 0b_calcular_pesos.py sobre los tiles de
# TRAIN vigentes, incluida la campana 2017-2018 (raiz del inverso de frecuencia).
# Orden: [NoData, Fondo, Maiz, Soja, Mani]
PESOS = [0.0, 0.742, 1.119, 0.868, 4.105]
# Sin la campana 2017-2018: los pesos previos, calculados sobre los 13 tiles
# 2019-2020 (las etiquetas no dependen de la ventana de meses).
if ARGS.sin_2017:
    PESOS = [0.0, 0.851, 1.019, 0.79, 4.245]
if ARGS.pesos:
    PESOS = ARGS.pesos

NOMBRES = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani"}

torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
# TileDatasetMmap (utils/tile_dataset.py) lee cada parche del disco bajo
# demanda via mmap en vez de cargar el tile entero en RAM. El flag
# remap_aplicado que antes se validaba aca al cargar el .npz ahora se valida
# una sola vez, al convertir con 03b_convertir_datasets_a_mmap.py -- si el
# .npy existe es porque ya paso esa verificacion. Tambien aplica la
# augmentacion (flips + rotaciones de 90) cuando augment=True, solo para TRAIN.

def cargar_tiles(tiles, etiqueta, augment=False, dir_train=None):
    print(f"\nCargando {etiqueta}...", flush=True)
    datasets = []
    canales = set()
    for t in tiles:
        ruta_base = f"{dir_train or DIR_TRAIN}/{PREFIJO}{t}"
        if not os.path.exists(f"{ruta_base}_X.npy"):
            print(f"  {t}: NO EXISTE, salteado", flush=True)
            continue
        ds = TileDatasetMmap(ruta_base, augment=augment)

        # Lectura unica y transitoria (Y es uint8, liviano) para validar el
        # rango de clases -- a diferencia de X, no queda residente en RAM
        # durante todo el entrenamiento, ds sigue mmapeado.
        clase_max = int(np.asarray(ds.Y).max())
        if clase_max >= NUM_CLASSES:
            raise ValueError(f"{ruta_base}: clase {clase_max} pero NUM_CLASSES={NUM_CLASSES}")

        canales.add(ds.X.shape[1])
        datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"No se cargo ningun tile para {etiqueta}")
    if len(canales) > 1:
        raise RuntimeError(f"Canales inconsistentes en {etiqueta}: {canales}")

    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


# ══════════════════════════════════════════════════════════════════════════════
# METRICAS
# ══════════════════════════════════════════════════════════════════════════════
def correctos_total(preds, labels, ignore_index=0):
    pred = torch.argmax(preds, dim=1)
    mask = labels != ignore_index
    return (pred[mask] == labels[mask]).sum(), mask.sum()


def evaluar(modelo, loader, criterion, device):
    modelo.eval()
    loss_sum = torch.zeros((), device=device)
    ok_sum   = torch.zeros((), device=device)
    tot_sum  = torch.zeros((), device=device)
    n = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        for X, Y in loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            out = modelo(X)
            loss_sum += criterion(out, Y)
            ok, tot = correctos_total(out, Y)
            ok_sum += ok; tot_sum += tot; n += 1
    total = tot_sum.item()
    return (loss_sum / n).item(), (ok_sum.item() / total if total else 0.0)


def chequeo_sanidad(modelo, loader, device):
    """Verifica que el modelo no prediga una sola clase (sintoma de bug)."""
    modelo.eval()
    X, Y = next(iter(loader))
    with torch.no_grad():
        pred = torch.argmax(modelo(X[:8].to(device)), dim=1).cpu()
    cp = torch.unique(pred).tolist()
    cr = torch.unique(Y[:8]).tolist()
    print(f"\n  Chequeo de sanidad — reales: {cr} | predichas: {cp}", flush=True)
    if len(cp) == 1:
        print("  ADVERTENCIA: el modelo predice una sola clase", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def guardar_json(ruta, datos):
    temporal = ruta.with_suffix('.json.tmp')
    temporal.write_text(json.dumps(datos, indent=2, ensure_ascii=False))
    temporal.replace(ruta)


def evaluar_confusion(modelo, loader, device):
    modelo.eval()
    cm = torch.zeros(25, dtype=torch.int64, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        for x, y in loader:
            y = y.to(device, non_blocking=True)
            pred = modelo(x.to(device, non_blocking=True)).argmax(1)
            validos = y != 0
            cm += torch.bincount(y[validos] * 5 + pred[validos], minlength=25)
    return cm.reshape(5, 5).cpu().numpy()

def entrenar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Dispositivo: {device}", flush=True)

    global PESOS
    por_campana = None
    informe = None
    if ARGS.campana:
        campanas, informe, conteos = validar_campanas(ARGS.campana)
        PESOS = ARGS.pesos if ARGS.pesos is not None else pesos_desde_conteos(conteos)
        print(f"Conteos TRAIN: {conteos.tolist()} | Pesos: {PESOS}", flush=True)
        if ARGS.verificar_datos:
            print("VERIFICACION OK; no se entreno ni se crearon salidas", flush=True)
            return
        conjuntos, por_campana = cargar_campanas(campanas)
        ds_train, ds_val, ds_test = (conjuntos[r] for r in ('train', 'val', 'test'))
    else:
        ds_train = cargar_tiles(TILES_TRAIN, "TRAIN", augment=True)
        if ARGS.dir_train_extra:
            # Los tiles _2017 son de train_cordoba_f16 (oct-abr), no de este dataset
            ds_extra = cargar_tiles([t for t in TILES_TRAIN if not t.endswith("_2017")],
                                    f"TRAIN extra ({ARGS.dir_train_extra})", augment=True,
                                    dir_train=ARGS.dir_train_extra)
            ds_train = ConcatDataset([ds_train, ds_extra])
            if ds_train[0][0].shape != ds_extra[0][0].shape:
                raise RuntimeError("Canales distintos entre --dir-train y --dir-train-extra")
        ds_val   = cargar_tiles(TILE_VAL,    "VAL")
        ds_test  = cargar_tiles(TILES_TEST,  "TEST")

    os.makedirs(DIR_EXP, exist_ok=True)
    if informe is not None:
        guardar_json(Path(DIR_EXP) / 'config_multicampana.json', {
            'campanas': informe, 'split': SPLIT, 'pesos': PESOS,
            'conteos_train': conteos.tolist(), 'modelo': ARGS.modelo,
            'ventana': 'diciembre-mayo', 'bandas': ['B02', 'B03', 'B04', 'B08'],
            'canales': 24, 'seed': SEED,
            'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'patience': PATIENCE,
            'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        })

    print(f"Pesos por clase: {PESOS}", flush=True)
    print(f"\nTrain: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}", flush=True)

    # Con TileDatasetMmap cada __getitem__ hace I/O real (lee del disco bajo
    # demanda), asi que ahora si vale la pena paralelizar con varios workers:
    # se solapan con el computo en GPU en vez de dejarla esperando.
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
              pin_memory=True, persistent_workers=True,
              prefetch_factor=PREFETCH_FACTOR)
    train_loader = DataLoader(ds_train, shuffle=True,  **kw)
    val_loader   = DataLoader(ds_val,   shuffle=False, **kw)
    test_loader  = DataLoader(ds_test,  shuffle=False, **kw)

    IN_CH = ds_train[0][0].shape[0]
    print(f"Canales de entrada: {IN_CH}", flush=True)
    print(f"Modelo: {ARGS.modelo}", flush=True)

    if ARGS.modelo == "unet":
        model = SimpleUNet(IN_CH, NUM_CLASSES).to(device)
    elif ARGS.modelo == "transunet":
        model = TransUNet(IN_CH, NUM_CLASSES).to(device)
    elif ARGS.modelo == "utae":
        model = UTAE(IN_CH, NUM_CLASSES, T=IN_CH // 4).to(device)
    else:
        model = FCRTransUNet(IN_CH, NUM_CLASSES).to(device)
    pesos = torch.tensor(PESOS, dtype=torch.float32).to(device)
    # CE (con pesos) + Dice: Dice refuerza la superposicion de Mani, que CE
    # subestima aun con pesos altos por ser una fraccion muy chica de los
    # pixeles utiles.
    criterion = CEDiceLoss(weight=pesos, num_classes=NUM_CLASSES, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5)
    # Mixed precision: aprovecha los Tensor Cores de la RTX 3090. Con la U-Net
    # de 4 niveles (mas pesada que la version chica original) esto reduce
    # computo real, no solo overhead de lanzar kernels.
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    chequeo_sanidad(model, train_loader, device)

    mejor = float("inf")
    sin_mejora = 0
    historial = []

    print("\n--- Iniciando entrenamiento ---", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = torch.zeros((), device=device)
        ok_sum   = torch.zeros((), device=device)
        tot_sum  = torch.zeros((), device=device)
        n = 0

        for X, Y in train_loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(X)
                loss = criterion(out, Y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                loss_sum += loss.detach()
                ok, tot = correctos_total(out, Y)
                ok_sum += ok; tot_sum += tot
            n += 1

        tr_loss = (loss_sum / n).item()
        total = tot_sum.item()
        tr_acc = ok_sum.item() / total if total else 0.0
        va_loss, va_acc = evaluar(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        historial.append([epoch+1, tr_loss, tr_acc, va_loss, va_acc])
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | "
              f"Val Loss:{va_loss:.4f} Acc:{va_acc:.4f} | lr={lr:.1e}", flush=True)

        # Alarma temprana
        if epoch == 0 and tr_acc > 0.95:
            print("  ADVERTENCIA: accuracy >95% en epoca 1 — revisar etiquetas", flush=True)

        if va_loss < mejor:
            mejor = va_loss
            torch.save(model.state_dict(), f"{DIR_EXP}/best_model.pth")
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= PATIENCE:
                print(f"Early stopping en epoca {epoch+1}", flush=True)
                break

    np.save(f"{DIR_EXP}/historial.npy", np.array(historial))
    torch.save(model.state_dict(), f"{DIR_EXP}/modelo_final.pth")

    print("\n--- Evaluando en TEST con el mejor modelo ---", flush=True)
    model.load_state_dict(torch.load(f"{DIR_EXP}/best_model.pth"))
    te_loss, te_acc = evaluar(model, test_loader, criterion, device)
    print(f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f}", flush=True)
    if por_campana is not None:
        resultados = {}
        # Sin workers persistentes: cada loader temporal se libera tras su evaluacion.
        for rol in ('val', 'test'):
            suma = np.zeros((5, 5), dtype=np.int64)
            resultados[rol] = {}
            for nombre, conjuntos_anio in por_campana.items():
                loader = DataLoader(conjuntos_anio[rol], batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=2, pin_memory=True)
                cm = evaluar_confusion(model, loader, device)
                suma += cm
                resultados[rol][nombre] = metricas_confusion(cm)
                print(f"{rol.upper()} {nombre}: {resultados[rol][nombre]}", flush=True)
            resultados[rol]['global'] = metricas_confusion(suma)
        guardar_json(Path(DIR_EXP) / 'metricas_multicampana.json', resultados)
    print(f"\nResultados en: {DIR_EXP}", flush=True)


if __name__ == "__main__":
    entrenar()
