"""
Compara los 4 modelos entrenados (unet, transunet, utae, fcrtransunet) en un
unico lugar:
  - Curva de error: Val Loss / Val Accuracy por epoca, superpuestas (un color
    por modelo), leidas del historial.npy que ya guarda cada entrenamiento.
  - Mapa de comparacion: Target + prediccion de cada modelo, lado a lado,
    sobre un tile (por default uno de TEST, para que sea evaluacion real).

Uso:
    python3 src/8_comparar_modelos.py --tile 20HLJ
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys, os
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet, TransUNet, UTAE, FCRTransUNet

BASE_DIR = "/mnt/yacy_1/prod/ferreyra/dataset"
# Mismo orden en el que se fueron probando en la sesion (ver docstrings de
# utils/model.py para el porque de cada uno).
MODELOS = {
    "unet":         (SimpleUNet,    f"{BASE_DIR}/exp_cordoba_f16"),
    "transunet":    (TransUNet,     f"{BASE_DIR}/exp_cordoba_transunet_f16"),
    "utae":         (UTAE,          f"{BASE_DIR}/exp_cordoba_utae_f16"),
    "fcrtransunet": (FCRTransUNet,  f"{BASE_DIR}/exp_cordoba_fcrtransunet_f16"),
}
COLORES_MODELO = {"unet": "tab:gray", "transunet": "tab:orange",
                  "utae": "tab:red", "fcrtransunet": "tab:blue"}
NOMBRES_MODELO = {"unet": "UNet (baseline)", "transunet": "TransUNet",
                  "utae": "UTAE", "fcrtransunet": "FCR-TransUNet"}

# Misma paleta que 6_generar_comparacion_visual.py (esquema vigente, 5 clases sin Sorgo)
COLORES_CLASE = {0: [0, 0, 0], 1: [70, 130, 180], 2: [255, 165, 0], 3: [0, 100, 0], 4: [165, 42, 42]}
NOMBRES_CLASE = {1: "Fondo", 2: "Maiz", 3: "Soja", 4: "Mani"}


def clase_a_rgb(mapa):
    img = np.zeros((*mapa.shape, 3), dtype=np.uint8)
    for c, color in COLORES_CLASE.items():
        img[mapa == c] = color
    return img


def graficar_curvas(salida):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curva de error por epoca — comparacion de modelos", fontsize=13)
    for nombre, (_, dir_exp) in MODELOS.items():
        ruta = Path(dir_exp) / "historial.npy"
        if not ruta.exists():
            print(f"  {nombre}: sin historial.npy, salteado")
            continue
        h = np.load(ruta)  # columnas: epoca, train_loss, train_acc, val_loss, val_acc
        color = COLORES_MODELO[nombre]
        axes[0].plot(h[:, 0], h[:, 3], color=color, label=NOMBRES_MODELO[nombre], linewidth=1.6)
        axes[1].plot(h[:, 0], h[:, 4], color=color, label=NOMBRES_MODELO[nombre], linewidth=1.6)

    axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Val Loss")
    axes[0].set_title("Val Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Val Accuracy")
    axes[1].set_title("Val Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {salida}")


def graficar_mapa(tile, dir_train, prefijo, n_clases, salida):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(f"{dir_train}/{prefijo}{tile}.npz")
    X_all, Y_all = data["X"], data["Y"]

    n_parches = len(X_all)
    lado = int(np.sqrt(n_parches))
    n_usar = lado * lado
    X_usar, Y_usar = X_all[:n_usar], Y_all[:n_usar]
    size = 256

    mapa_target = np.zeros((lado * size, lado * size), dtype=np.int64)
    for i in range(lado):
        for j in range(lado):
            mapa_target[i*size:(i+1)*size, j*size:(j+1)*size] = Y_usar[i*lado+j]

    paneles = [("Target (real)", mapa_target)]
    for nombre, (clase, dir_exp) in MODELOS.items():
        ckpt = Path(dir_exp) / "best_model.pth"
        if not ckpt.exists():
            print(f"  {nombre}: sin checkpoint, salteado")
            continue
        modelo = clase(28, n_clases).to(device)
        modelo.load_state_dict(torch.load(ckpt, map_location=device))
        modelo.eval()

        preds = []
        for i in range(0, n_usar, 16):
            # .float(): X viene en float16 (ahorro de RAM), el modelo esta en float32.
            xb = torch.from_numpy(X_usar[i:i+16]).float().to(device)
            with torch.no_grad():
                pred = torch.argmax(modelo(xb), dim=1).cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)

        mapa_pred = np.zeros((lado * size, lado * size), dtype=np.int64)
        for i in range(lado):
            for j in range(lado):
                mapa_pred[i*size:(i+1)*size, j*size:(j+1)*size] = preds[i*lado+j]
        paneles.append((NOMBRES_MODELO[nombre], mapa_pred))

        del modelo
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, len(paneles), figsize=(5 * len(paneles), 6))
    fig.suptitle(f"Comparacion de modelos — Tile {tile}", fontsize=14)
    for ax, (titulo, mapa) in zip(axes, paneles):
        ax.imshow(clase_a_rgb(mapa))
        ax.set_title(titulo)
        ax.axis('off')

    leyenda = [Patch(color=[c / 255 for c in COLORES_CLASE[k]], label=v) for k, v in NOMBRES_CLASE.items()]
    fig.legend(handles=leyenda, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig(salida, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="20HLJ", help="Tile para el mapa de comparacion (default: uno de TEST)")
    parser.add_argument("--dir_train", default="/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16")
    parser.add_argument("--prefijo", default="dataset_")
    parser.add_argument("--n_clases", type=int, default=5)
    parser.add_argument("--salida_dir", default=".")
    args = parser.parse_args()

    graficar_curvas(f"{args.salida_dir}/comparacion_curvas.png")
    graficar_mapa(args.tile, args.dir_train, args.prefijo, args.n_clases,
                  f"{args.salida_dir}/comparacion_mapa_{args.tile}.png")
