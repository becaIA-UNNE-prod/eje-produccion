"""
Entrenamiento de clasificacion de cultivos de verano sobre Cordoba completa.
Dataset: composites Sentinel-2 2019-2020 + etiquetas MNC INTA verano 2020.

Los .npz ya tienen el remapeo a 6 clases aplicado (flag remap_aplicado=1),
por eso NO se vuelve a remapear al cargar.

Uso:
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = "/mnt/yacy_1/prod/ferreyra/dataset"
DIR_TRAIN  = f"{BASE_DIR}/train_cordoba_f16"
DIR_EXP    = f"{BASE_DIR}/exp_cordoba_aug"
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
class TileDataset(torch.utils.data.Dataset):
    """Carga un .npz completo en RAM. NO aplica remapeo.

    Si augment=True aplica flips y rotaciones aleatorias de 90 grados.
    Son transformaciones validas para imagenes satelitales: un lote de soja
    sigue siendo soja si se rota o se espeja. Solo en TRAIN, nunca en val/test."""

    def __init__(self, ruta_npz, augment=False):
        self.augment = augment
        data = np.load(ruta_npz)

        # Verificar que el remapeo ya fue aplicado al generar el archivo
        if "remap_aplicado" not in data:
            raise ValueError(
                f"{ruta_npz} no tiene el flag remap_aplicado. "
                "Regenerar el dataset o verificar el esquema de clases.")

        # float16 en RAM (mitad de memoria); se convierte a float32 por
        # parche en __getitem__, que es lo que espera PyTorch en las convs.
        self.X = torch.from_numpy(data["X"])
        self.Y = torch.from_numpy(data["Y"].astype(np.int64))

        clases = torch.unique(self.Y).tolist()
        if max(clases) >= NUM_CLASSES:
            raise ValueError(
                f"{ruta_npz} tiene clase {max(clases)} pero NUM_CLASSES={NUM_CLASSES}")

        print(f"  {os.path.basename(ruta_npz)}: {len(self.X)} parches, "
              f"clases {clases}", flush=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].float()
        y = self.Y[i]
        if self.augment:
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
                y = torch.rot90(y, k, dims=(0, 1))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(2,))
                y = torch.flip(y, dims=(1,))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(1,))
                y = torch.flip(y, dims=(0,))
        return x.contiguous(), y.contiguous()


def cargar_tiles(tiles, etiqueta, augment=False):
    print(f"\nCargando {etiqueta}...", flush=True)
    datasets = []
    canales = set()
    for t in tiles:
        ruta = f"{DIR_TRAIN}/{PREFIJO}{t}.npz"
        if not os.path.exists(ruta):
            print(f"  {t}: NO EXISTE, salteado", flush=True)
            continue
        ds = TileDataset(ruta, augment=augment)
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
    with torch.no_grad():
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

    ds_train = cargar_tiles(TILES_TRAIN, "TRAIN", augment=True)
    ds_val   = cargar_tiles(TILE_VAL,    "VAL")
    ds_test  = cargar_tiles(TILES_TEST,  "TEST")

    print(f"\nTrain: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}", flush=True)

    kw = dict(batch_size=BATCH_SIZE, num_workers=2,
              pin_memory=True, persistent_workers=True)
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
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
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
