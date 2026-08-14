"""
Genera la matriz de confusion sobre un tile de evaluacion.

Uso:
    python3 src/5_generar_matriz_confusion.py \
        --exp /mnt/yacy_1/prod/ferreyra/dataset/exp_verano2 \
        --tile 20HNK \
        --dir_train /mnt/yacy_1/prod/ferreyra/dataset/train
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import sys, os
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True)
parser.add_argument('--tile', required=True)
parser.add_argument('--dir_train', required=True)
parser.add_argument('--n_clases', type=int, default=6)
parser.add_argument('--prefijo', default='dataset_T')
args = parser.parse_args()

DIR_EXP = Path(args.exp)
REMAP_6 = np.array([0, 1, 1, 1, 2, 3, 4, 5, 1], dtype=np.int64)
NOMBRES_6 = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani", 5:"Sorgo"}
# Esquema original de 9 clases (mismo orden que LABEL_REMAP en 03_generar_datasets_npz.py)
NOMBRES_9 = {0:"NoData", 1:"Natural", 2:"Urbano", 3:"Trigo", 4:"Maiz",
             5:"Soja", 6:"Mani", 7:"Sorgo", 8:"Otros"}
NOMBRES = NOMBRES_6 if args.n_clases == 6 else NOMBRES_9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet(28, args.n_clases).to(device)
model.load_state_dict(torch.load(DIR_EXP / 'best_model.pth'))
model.eval()

data = np.load(f"{args.dir_train}/{args.prefijo}{args.tile}.npz")
X_all = data["X"]
# Remapear a 6 clases solo si el target son 6 clases y el .npz trae las 9 originales
# sin curar (dataset_T*). Si se pide evaluar con 9 clases (baseline), dejar sin tocar.
Y_all = REMAP_6[data["Y"]] if (args.prefijo == "dataset_T" and args.n_clases == 6) else data["Y"]

confusion = np.zeros((args.n_clases, args.n_clases), dtype=np.int64)
for i in range(0, len(X_all), 16):
    X_b = torch.from_numpy(X_all[i:i+16]).to(device)
    Y_b = Y_all[i:i+16]
    with torch.no_grad():
        pred = torch.argmax(model(X_b), dim=1).cpu().numpy()
    for r, p in zip(Y_b.flatten(), pred.flatten()):
        if r > 0:
            confusion[r][p] += 1

clases = [i for i in range(1, args.n_clases) if confusion[i].sum() > 0]
nombres = [NOMBRES[i] for i in clases]
conf_norm = np.zeros((len(clases), len(clases)))
for i, c in enumerate(clases):
    if confusion[c].sum() > 0:
        conf_norm[i] = confusion[c, clases] / confusion[c].sum()

total_ok = sum(confusion[c][c] for c in clases)
total = sum(confusion[c].sum() for c in clases)
acc_global = total_ok / total * 100 if total > 0 else 0

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(clases))); ax.set_yticks(range(len(clases)))
ax.set_xticklabels(nombres, rotation=45, ha='right')
ax.set_yticklabels(nombres)
ax.set_xlabel("Predicha"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusión — Tile {args.tile}\nAccuracy global: {acc_global:.1f}%")

for i in range(len(clases)):
    for j in range(len(clases)):
        val = conf_norm[i, j]
        count = confusion[clases[i], clases[j]]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{count:,}\n({val:.1%})", ha="center", va="center",
                color=color, fontsize=9)

plt.tight_layout()
ruta = DIR_EXP / f'matriz_confusion_{args.tile}.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {ruta}")
