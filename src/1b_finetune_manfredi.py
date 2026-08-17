import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet
from utils.tile_dataset import TileDatasetMmap
from utils.losses import CEDiceLoss

# ============================================================
# CONFIGURACION
# ============================================================
BASE_DIR = "/mnt/yacy_1/prod/ferreyra/dataset"

# Checkpoint ya entrenado (1_train_cordoba.py, Cordoba completa) que se va a
# especializar en Manfredi.
RUTA_CHECKPOINT_BASE = f"{BASE_DIR}/exp_cordoba_f16/best_model.pth"

# Deben coincidir con los usados para generar RUTA_CHECKPOINT_BASE (mismos meses/canales).
DIR_TRAIN      = f"{BASE_DIR}/train_cordoba_f16"
DATASET_PREFIX = "dataset_"

# Completar con el resultado de src/buscar_tile_por_coordenada.py
TILE_MANFREDI = "20HMK"

DIR_EXP = f"{BASE_DIR}/exp_manfredi_ft"

# 5 clases: 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani (sin Sorgo, ver README_MNC.md)
NUM_CLASSES   = 5
BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-5  # bajo: fine-tuning sobre un modelo ya entrenado, no entrenamiento desde cero
PATIENCE      = 10
FRACCION_VAL  = 0.15
SEED          = 42

# floryacy: 32 nucleos de CPU / RTX 3090. Con TileDatasetMmap cada __getitem__
# lee del disco bajo demanda (antes el dataset ya estaba entero en RAM), asi
# que varios workers en paralelo si aportan (se solapan con el computo en GPU).
NUM_WORKERS = 16
PREFETCH_FACTOR = 4

os.makedirs(DIR_EXP, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# METRICAS
# ============================================================
def calcular_correctos_total(preds, labels, ignore_index=0):
    pred_clases = torch.argmax(preds, dim=1)
    mascara = labels != ignore_index
    correctos = (pred_clases[mascara] == labels[mascara]).sum()
    total = mascara.sum()
    return correctos, total

def evaluar(modelo, loader, criterion, device):
    modelo.eval()
    loss_sum = torch.zeros((), device=device)
    correctos_sum = torch.zeros((), device=device)
    total_sum = torch.zeros((), device=device)
    n_batches = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        for X, Y in loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            out = modelo(X)
            loss_sum += criterion(out, Y)
            correctos, total = calcular_correctos_total(out, Y)
            correctos_sum += correctos
            total_sum += total
            n_batches += 1
    total_final = total_sum.item()
    acc = (correctos_sum.item() / total_final) if total_final > 0 else 0.0
    return (loss_sum / n_batches).item(), acc

# ============================================================
# FINE-TUNING
# ============================================================
def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    torch.backends.cudnn.benchmark = True

    if TILE_MANFREDI == "XXXXX":
        raise ValueError("Completar TILE_MANFREDI con el tile detectado por buscar_tile_por_coordenada.py")

    ruta_base = f"{DIR_TRAIN}/{DATASET_PREFIX}{TILE_MANFREDI}"
    print(f"\nCargando parches de Manfredi (tile {TILE_MANFREDI})...")
    dataset = TileDatasetMmap(ruta_base)

    n = len(dataset)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = max(1, int(n * FRACCION_VAL))
    idx_val, idx_train = idx[:n_val], idx[n_val:]

    ds_train = Subset(dataset, idx_train)
    ds_val = Subset(dataset, idx_val)
    print(f"Train: {len(ds_train)} parches | Val: {len(ds_val)} parches")

    # Con TileDatasetMmap cada __getitem__ hace I/O real (lee del disco bajo
    # demanda), asi que ahora si vale la pena paralelizar con varios workers:
    # se solapan con el computo en GPU en vez de dejarla esperando.
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          pin_memory=True, persistent_workers=True,
                          prefetch_factor=PREFETCH_FACTOR)
    train_loader = DataLoader(ds_train, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False, **loader_kwargs)

    IN_CHANNELS = dataset[0][0].shape[0]
    print(f"Canales de entrada: {IN_CHANNELS}")

    model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(RUTA_CHECKPOINT_BASE, map_location=device))

    # Mismos pesos y loss (CE+Dice) que 1_train_cordoba.py, para no cambiar el
    # objetivo de optimizacion entre el entrenamiento base y el fine-tuning.
    pesos = torch.tensor([0.0, 0.716, 0.947, 0.728, 4.545], dtype=torch.float32).to(device)
    criterion = CEDiceLoss(weight=pesos, num_classes=NUM_CLASSES, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Mixed precision: aprovecha los Tensor Cores de la RTX 3090. Con la U-Net
    # de 4 niveles (mas pesada que la version chica original) esto reduce
    # computo real, no solo overhead de lanzar kernels.
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    mejor_val_loss = float("inf")
    epocas_sin_mejora = 0
    historial = []

    print("\n--- Iniciando fine-tuning sobre Manfredi ---")
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = torch.zeros((), device=device)
        correctos_sum = torch.zeros((), device=device)
        total_sum = torch.zeros((), device=device)
        n_batches = 0

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
                correctos, total = calcular_correctos_total(out, Y)
                correctos_sum += correctos
                total_sum += total
            n_batches += 1

        train_loss = (loss_sum / n_batches).item()
        total_final = total_sum.item()
        train_acc = (correctos_sum.item() / total_final) if total_final > 0 else 0.0
        val_loss, val_acc = evaluar(model, val_loader, criterion, device)

        historial.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f}")

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(model.state_dict(), f"{DIR_EXP}/best_model.pth")
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PATIENCE:
                print(f"Early stopping en época {epoch+1}")
                break

    np.save(f"{DIR_EXP}/historial.npy", np.array(historial))
    print(f"\nResultados guardados en: {DIR_EXP}")

if __name__ == "__main__":
    finetune()

