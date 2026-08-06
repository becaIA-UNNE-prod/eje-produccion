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

# Dataset generado por 03_generar_datasets_npz.py (dataset_T<TILE>.npz, 9 clases sin remapear).
# Si en algún momento se cuenta con un dataset ya curado a 6 clases (ej. dataset_mnc_T<TILE>.npz,
# generado por un proceso externo a este repo), cambiar a:
#   DIR_TRAIN, DIR_EXP, DATASET_PREFIX = f"{BASE_DIR}/train_mnc", f"{BASE_DIR}/exp_mnc", "dataset_mnc_T"
#   APLICAR_REMAP_6 = False
DIR_TRAIN       = f"{BASE_DIR}/train_multitile7"
DIR_EXP         = f"{BASE_DIR}/exp_multitile7"
DATASET_PREFIX  = "dataset_T"
APLICAR_REMAP_6 = True

# 6 tiles (frente a los 3+1+1 anteriores) con 7 meses en común (201707-201712,
# 201804) generados por 03_generar_datasets_npz.py con MESES_COMUNES. Mismo
# val/test que la config anterior para mantener comparabilidad. (19HGB se
# descartó: 0 parches válidos, máscara sin datos de cultivo en esa zona.)
TILES_TRAIN = ["20HLJ", "20HLK", "20JLL", "20JML"]
TILE_VAL    = "20JNL"
TILES_TEST  = ["20HNK"]

# Modelo chico (UNet de 2 niveles) -> con batch=16 la GPU pasa la mayor parte
# del tiempo ociosa. Con una RTX 3090 (24GB) 32 deja bastante margen; si el
# uso de VRAM (nvidia-smi) queda bajo, se puede subir a 48/64.
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 1e-4
NUM_CLASSES   = 6
PATIENCE      = 50
SEED          = 42

# Remapeo de 9 clases originales a solo cultivos de verano (6 clases):
# 0=NoData,1=Natural,2=Urbano,3=Trigo,4=Maiz,5=Soja,6=Mani,7=Sorgo,8=Otros
# ->        0=NoData,1=Fondo, 1=Fondo,1=Fondo,2=Maiz,3=Soja,4=Mani,5=Sorgo,1=Fondo
REMAP_6 = np.array([0, 1, 1, 1, 2, 3, 4, 5, 1], dtype=np.int64)

os.makedirs(DIR_EXP, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# DATASET DESDE .NPZ
# ============================================================
class TileDataset(torch.utils.data.Dataset):
    def __init__(self, ruta_npz, fraccion=1.0, seed=42, aplicar_remap=False):
        data = np.load(ruta_npz)
        X = data["X"]
        Y = data["Y"]
        if aplicar_remap:
            Y = REMAP_6[Y]
        if fraccion < 1.0:
            # Solo copiamos/barajamos si realmente vamos a submuestrear: con
            # fraccion=1.0 el fancy-indexing duplicaba el tile entero en RAM
            # innecesariamente (X[idx] con idx = todos los índices barajados).
            n = int(len(X) * fraccion)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=n, replace=False)
            X, Y = X[idx], Y[idx]
        else:
            n = len(X)
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.Y = torch.from_numpy(Y.astype(np.int64, copy=False))
        print(f"  {os.path.basename(ruta_npz)}: {n} parches cargados")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# ============================================================
# METRICAS
# ============================================================
def calcular_correctos_total(preds, labels, ignore_index=0):
    # Devuelve tensores en GPU (sin .item()) para no forzar una sincronización
    # CPU/GPU en cada batch: eso serializa el loop y deja la GPU ociosa esperando.
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
# ENTRENAMIENTO
# ============================================================
def entrenar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    # Todas las imágenes son de 256x256 -> forma fija de entrada, cudnn puede
    # elegir el algoritmo de convolución más rápido para esa forma sin volver
    # a probar en cada batch.
    torch.backends.cudnn.benchmark = True

    print("\nCargando tiles de TRAIN...")
    ds_train = ConcatDataset([
        TileDataset(f"{DIR_TRAIN}/{DATASET_PREFIX}{t}.npz", aplicar_remap=APLICAR_REMAP_6) for t in TILES_TRAIN
    ])

    print("\nCargando tile de VAL...")
    ds_val = TileDataset(f"{DIR_TRAIN}/{DATASET_PREFIX}{TILE_VAL}.npz", fraccion=1.0, aplicar_remap=APLICAR_REMAP_6)

    print("\nCargando tiles de TEST...")
    ds_test = ConcatDataset([
        TileDataset(f"{DIR_TRAIN}/{DATASET_PREFIX}{t}.npz", aplicar_remap=APLICAR_REMAP_6) for t in TILES_TEST
    ])

    print(f"\nTrain: {len(ds_train)} parches | Val: {len(ds_val)} | Test: {len(ds_test)}")

    # El dataset ya está entero en RAM (TileDataset lo carga en __init__), así que
    # __getitem__ es solo indexar tensores: num_workers>0 aporta poco y pin_memory
    # + non_blocking permiten que la copia CPU->GPU se solape con el cómputo.
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2,
                               pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                               pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                               pin_memory=True, persistent_workers=True)

    IN_CHANNELS = ds_train[0][0].shape[0]
    print(f"Canales de entrada: {IN_CHANNELS}")

    model     = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    # Pesos suavizados (raiz cuadrada del inverso de frecuencia)
    # para combatir el desbalance de clases en el dataset de train
    # Pesos para 6 clases: 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani, 5=Sorgo
    pesos = torch.tensor([0.0, 0.3, 1.0, 0.8, 3.0, 8.0],
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=True
    )
    mejor_val_loss = float("inf")
    epocas_sin_mejora = 0
    historial = []

    print("\n--- Iniciando entrenamiento ---")
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
            out  = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            # Acumular en GPU y recién bajar a Python al final de la época:
            # un .item() por batch fuerza sincronización CPU/GPU en cada paso
            # y es lo que dejaba la GPU esperando en vez de trabajando.
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
        scheduler.step(val_loss)

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
