"""
Entrenamiento de clasificacion de cultivos de verano sobre Cordoba completa.
Dataset: composites Sentinel-2 2019-2020 + etiquetas MNC INTA verano 2020.

Los datos ya tienen el remapeo a 6 clases aplicado, por eso NO se vuelve a
remapear al cargar. Requiere haber convertido los .npz de origen a pares
<tile>_X.npy/_Y.npy sueltos (ver src/03b_convertir_datasets_a_mmap.py), que
es donde se valida una sola vez el flag remap_aplicado=1 de los .npz.

Uso:
    python3 src/03b_convertir_datasets_a_mmap.py /mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16
    python3 src/1_train_cordoba.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet
from utils.tile_dataset import TileDatasetMmap

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = "/mnt/yacy_1/prod/ferreyra/dataset"
DIR_TRAIN  = f"{BASE_DIR}/train_cordoba_f16"
DIR_EXP    = f"{BASE_DIR}/exp_cordoba_f16"
PREFIJO    = "dataset_"

# Split por franjas de cobertura de cultivo (ver analisis de cobertura MNC)
# Split espacial por tiles completos, balanceado en tres criterios:
#   1) gradiente geografico oeste-centro-este (evita mismatch de dominio)
#   2) presencia de Mani (clase minoritaria) en los tres conjuntos
#   3) validacion ampliada a 4 tiles para reducir el ruido de la curva Val
# Split de 11 tiles (limitado por RAM disponible: 124 GB).
# Mantiene cobertura oeste-centro-este y Mani en los 3 conjuntos.
TILES_TRAIN = ['19HGC',
               '20HLG','20HLH','20HMJ',
               '20HMK','20HNK','20JML']
TILE_VAL    = ['20HKJ','20HMH']
TILES_TEST  = ['20HLK','20HLJ']

NUM_CLASSES   = 5
BATCH_SIZE    = 32
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

# Pesos por clase — recalcular con 0_verificar_pipeline.py --etapa pesos
# Orden: [NoData, Fondo, Maiz, Soja, Mani, Sorgo]
PESOS = [0.0, 0.716, 0.947, 0.728, 4.545]

NOMBRES = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani"}

os.makedirs(DIR_EXP, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
# TileDatasetMmap (utils/tile_dataset.py) lee cada parche del disco bajo
# demanda via mmap en vez de cargar el tile entero en RAM. El flag
# remap_aplicado que antes se validaba aca al cargar el .npz ahora se valida
# una sola vez, al convertir con 03b_convertir_datasets_a_mmap.py -- si el
# .npy existe es porque ya paso esa verificacion.

def cargar_tiles(tiles, etiqueta):
    print(f"\nCargando {etiqueta}...", flush=True)
    datasets = []
    canales = set()
    for t in tiles:
        ruta_base = f"{DIR_TRAIN}/{PREFIJO}{t}"
        if not os.path.exists(f"{ruta_base}_X.npy"):
            print(f"  {t}: NO EXISTE, salteado", flush=True)
            continue
        ds = TileDatasetMmap(ruta_base)

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
def entrenar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Dispositivo: {device}", flush=True)

    ds_train = cargar_tiles(TILES_TRAIN, "TRAIN")
    ds_val   = cargar_tiles(TILE_VAL,    "VAL")
    ds_test  = cargar_tiles(TILES_TEST,  "TEST")

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

    model = SimpleUNet(IN_CH, NUM_CLASSES).to(device)
    pesos = torch.tensor(PESOS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos, ignore_index=0)
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
    print(f"\nResultados en: {DIR_EXP}", flush=True)


if __name__ == "__main__":
    entrenar()
