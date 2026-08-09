import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet

# ============================================================
# CONFIGURACION
# ============================================================
BASE_DIR = "/mnt/yacy_1/prod/ferreyra/dataset"

# Checkpoint ya entrenado (1_train_multitile.py) que se va a especializar en Manfredi.
RUTA_CHECKPOINT_BASE = f"{BASE_DIR}/exp_cordoba_mnc/best_model.pth"

# Deben coincidir con los usados para generar RUTA_CHECKPOINT_BASE (mismos meses/canales).
DIR_TRAIN      = f"{BASE_DIR}/train_cordoba_mnc"
DATASET_PREFIX = "dataset_mnc_"

# Completar con el resultado de src/buscar_tile_por_coordenada.py
TILE_MANFREDI = "20HMK"

DIR_EXP = f"{BASE_DIR}/exp_manfredi_ft"

NUM_CLASSES   = 6
BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-5  # bajo: fine-tuning sobre un modelo ya entrenado, no entrenamiento desde cero
PATIENCE      = 10
FRACCION_VAL  = 0.15
SEED          = 42

os.makedirs(DIR_EXP, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# DATASET DESDE .NPZ (un solo tile: Manfredi)
# ============================================================
class TileDataset(torch.utils.data.Dataset):
    def __init__(self, ruta_npz):
        data = np.load(ruta_npz)
        self.X = torch.from_numpy(data["X"].astype(np.float32))
        self.Y = torch.from_numpy(data["Y"].astype(np.int64))
        print(f"  {os.path.basename(ruta_npz)}: {len(self.X)} parches cargados")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

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
    with torch.no_grad():
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

    ruta_npz = f"{DIR_TRAIN}/{DATASET_PREFIX}{TILE_MANFREDI}.npz"
    print(f"\nCargando parches de Manfredi (tile {TILE_MANFREDI})...")
    dataset = TileDataset(ruta_npz)

    n = len(dataset)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = max(1, int(n * FRACCION_VAL))
    idx_val, idx_train = idx[:n_val], idx[n_val:]

    ds_train = Subset(dataset, idx_train)
    ds_val = Subset(dataset, idx_val)
    print(f"Train: {len(ds_train)} parches | Val: {len(ds_val)} parches")

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2,
                               pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                               pin_memory=True, persistent_workers=True)

    IN_CHANNELS = dataset[0][0].shape[0]
    print(f"Canales de entrada: {IN_CHANNELS}")

    model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(RUTA_CHECKPOINT_BASE, map_location=device))

    pesos = torch.tensor([0.0, 0.3, 1.0, 0.8, 3.0, 8.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

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

