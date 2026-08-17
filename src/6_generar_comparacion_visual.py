"""
Genera la imagen de comparacion visual: Target vs Prediccion vs Mapa de Errores.

Uso:
    python3 src/6_generar_comparacion_visual.py \
        --exp /mnt/yacy_1/prod/ferreyra/dataset/exp_verano2 \
        --tile 20JML \
        --dir_train /mnt/yacy_1/prod/ferreyra/dataset/train
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import sys, os
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True)
parser.add_argument('--tile', required=True)
parser.add_argument('--dir_train', required=True)
parser.add_argument('--n_clases', type=int, default=5)
parser.add_argument('--prefijo', default='dataset_')
args = parser.parse_args()

DIR_EXP = Path(args.exp)
REMAP_6 = np.array([0, 1, 1, 1, 2, 3, 4, 5, 1], dtype=np.int64)
COLORES = {0:[0,0,0], 1:[70,130,180], 2:[255,165,0],
           3:[0,100,0], 4:[165,42,42], 5:[255,0,0]}
# Esquema vigente (1_train_cordoba.py, Cordoba completa): 5 clases sin Sorgo
NOMBRES = {1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani"} if args.n_clases == 5 \
    else {1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani", 5:"Sorgo"}

def clase_a_rgb(mapa):
    img = np.zeros((*mapa.shape, 3), dtype=np.uint8)
    for clase, color in COLORES.items():
        img[mapa == clase] = color
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet(28, args.n_clases).to(device)
model.load_state_dict(torch.load(DIR_EXP / 'best_model.pth'))
model.eval()

data = np.load(f"{args.dir_train}/{args.prefijo}{args.tile}.npz")
X_all = data["X"]
Y_all = REMAP_6[data["Y"]] if args.prefijo == "dataset_T" else data["Y"]

n_parches = len(X_all)
lado = int(np.sqrt(n_parches))
n_usar = lado * lado
X_usar = X_all[:n_usar]
Y_usar = Y_all[:n_usar]

preds = []
for i in range(0, n_usar, 16):
    # .float(): train_cordoba_f16 guarda X en float16 (ahorro de RAM), pero el
    # modelo esta en float32 -- sin este cast, Conv2d tira dtype mismatch.
    X_b = torch.from_numpy(X_usar[i:i+16]).float().to(device)
    with torch.no_grad():
        pred = torch.argmax(model(X_b), dim=1).cpu().numpy()
    preds.append(pred)
preds = np.concatenate(preds, axis=0)

size = 256
mapa_target = np.zeros((lado*size, lado*size), dtype=np.int64)
mapa_pred   = np.zeros((lado*size, lado*size), dtype=np.int64)
for i in range(lado):
    for j in range(lado):
        idx = i * lado + j
        mapa_target[i*size:(i+1)*size, j*size:(j+1)*size] = Y_usar[idx]
        mapa_pred[i*size:(i+1)*size, j*size:(j+1)*size]   = preds[idx]

mapa_error = np.zeros((*mapa_target.shape, 3), dtype=np.uint8)
mascara = mapa_target > 1
mapa_error[mascara & (mapa_target == mapa_pred)] = [0, 200, 0]
mapa_error[mascara & (mapa_target != mapa_pred)] = [200, 0, 0]
mapa_error[~mascara] = [50, 50, 50]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(f"Target vs Predicción vs Mapa de Errores — Tile {args.tile}", fontsize=13)
axes[0].imshow(clase_a_rgb(mapa_target)); axes[0].set_title("Target (Ground Truth)"); axes[0].axis('off')
axes[1].imshow(clase_a_rgb(mapa_pred));   axes[1].set_title("Inferencia (Predicción)"); axes[1].axis('off')
axes[2].imshow(mapa_error);               axes[2].set_title("Mapa de Errores\n(Verde=Acierto, Rojo=Error, Gris=Fondo)"); axes[2].axis('off')

leyenda = [Patch(color=[c/255 for c in COLORES[k]], label=v) for k,v in NOMBRES.items()]
fig.legend(handles=leyenda, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
ruta = DIR_EXP / f'comparacion_{args.tile}.png'
plt.savefig(ruta, dpi=130, bbox_inches='tight')
plt.close()
print(f"Guardado: {ruta}")
