import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import sys
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet

# ============================================================
# CONFIGURACION
# ============================================================
BASE_DIR = "/mnt/yacy_1/prod/ferreyra/dataset"
DIR_TRAIN  = f"{BASE_DIR}/train_mnc"
DIR_EXP    = f"{BASE_DIR}/exp_mnc"

TILES_TRAIN = ["20HMJ", "20HMK", "20JML"]
TILE_VAL    = "20JNL"
TILES_TEST  = ["20HNK"]

BATCH_SIZE    = 16
EPOCHS        = 100
LEARNING_RATE = 1e-4
NUM_CLASSES   = 6
PATIENCE      = 50
SEED          = 42

os.makedirs(DIR_EXP, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# DATASET DESDE .NPZ
# ============================================================
class TileDataset(torch.utils.data.Dataset):
    def __init__(self, ruta_npz, fraccion=1.0, seed=42):
        data = np.load(ruta_npz)
        X = data["X"].astype(np.float32)
        Y = data["Y"].astype(np.int64)
        n = int(len(X) * fraccion)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n, replace=False)
        self.X = torch.from_numpy(X[idx])
        self.Y = torch.from_numpy(Y[idx])
        print(f"  {os.path.basename(ruta_npz)}: {n} parches cargados")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# ============================================================
# METRICAS
# ============================================================
def calcular_accuracy(preds, labels, ignore_index=0):
    pred_clases = torch.argmax(preds, dim=1)
    mascara = labels != ignore_index
    correctos = (pred_clases[mascara] == labels[mascara]).sum().item()
    total = mascara.sum().item()
    return correctos / total if total > 0 else 0.0

def evaluar(modelo, loader, criterion, device):
    modelo.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            out = modelo(X)
            total_loss += criterion(out, Y).item()
            total_acc  += calcular_accuracy(out, Y)
    return total_loss / len(loader), total_acc / len(loader)

# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print("\nCargando tiles de TRAIN...")
    ds_train = ConcatDataset([
        TileDataset(f"{DIR_TRAIN}/dataset_mnc_T{t}.npz") for t in TILES_TRAIN
    ])

    print("\nCargando tile de VAL...")
    ds_val = TileDataset(f"{DIR_TRAIN}/dataset_mnc_T{TILE_VAL}.npz", fraccion=1.0)

    print("\nCargando tiles de TEST...")
    ds_test = ConcatDataset([
        TileDataset(f"{DIR_TRAIN}/dataset_mnc_T{t}.npz") for t in TILES_TEST
    ])

    print(f"\nTrain: {len(ds_train)} parches | Val: {len(ds_val)} | Test: {len(ds_test)}")

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    IN_CHANNELS = ds_train[0][0].shape[0]
    print(f"Canales de entrada: {IN_CHANNELS}")

    model     = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)

    # Pesos por clase — 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani, 5=Sorgo
    pesos = torch.tensor([0.0, 0.3, 1.0, 0.8, 3.0, 8.0],
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    mejor_val_loss = float("inf")
    epocas_sin_mejora = 0
    historial = []

    print("\n--- Iniciando entrenamiento ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc  += calcular_accuracy(out, Y)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)
        val_loss, val_acc = evaluar(model, val_loader, criterion, device)

        historial.append([epoch+1, train_loss, train_acc, val_loss, val_acc])
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
    torch.save(model.state_dict(), f"{DIR_EXP}/modelo_final.pth")

    print("\n--- Evaluando en TEST con mejor modelo ---")
    model.load_state_dict(torch.load(f"{DIR_EXP}/best_model.pth"))
    test_loss, test_acc = evaluar(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"\nResultados guardados en: {DIR_EXP}")

if __name__ == "__main__":
    entrenar()
