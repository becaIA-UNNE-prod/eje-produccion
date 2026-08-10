"""
Genera las curvas de entrenamiento (Train/Val Loss y Accuracy)
a partir del historial.npy guardado en el directorio del experimento.

Uso:
    python3 src/4_generar_metricas.py --exp /mnt/yacy_1/prod/ferreyra/dataset/exp_verano2
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True, help='Directorio del experimento')
args = parser.parse_args()

DIR_EXP = Path(args.exp)
historial = np.load(DIR_EXP / 'historial.npy')
epocas = historial[:, 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Evolución del Entrenamiento — {DIR_EXP.name}", fontsize=12)

axes[0].plot(epocas, historial[:,1], 'b-', label='Train Loss', linewidth=1.5)
axes[0].plot(epocas, historial[:,3], 'r-', label='Val Loss', linewidth=1.5)
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
axes[0].set_title("Evolución de la Pérdida")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epocas, historial[:,2], 'b-', label='Train Acc', linewidth=1.5)
axes[1].plot(epocas, historial[:,4], 'r-', label='Val Acc', linewidth=1.5)
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Evolución de la Precisión")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
ruta = DIR_EXP / 'metricas.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {ruta}")
