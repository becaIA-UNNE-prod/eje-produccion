"""Matriz normalizada del evaluador MNC histórico, con nombres explícitos."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def graficar(cm, titulo, ruta, nombres):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), nombres); ax.set_yticks(range(4), nombres)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("MNC (referencia)")
    ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)
