"""Matrices de confusion y curvas de entrenamiento."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
NAMES = ["NoData","Fondo","Maiz","Soja","Mani"]
COLORS = np.array([[0,0,0],[70,130,180],[255,165,0],[0,100,0],[165,42,42]],dtype=np.uint8)

def matriz_fig(cm,title,path):
    norm = cm[1:,:]/np.maximum(cm[1:,:].sum(1,keepdims=True),1)
    fig,ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(4):
        for j in range(5): ax.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
    ax.set_xticks(range(5),NAMES); ax.set_yticks(range(4),NAMES[1:])
    ax.set_xlabel('Prediccion'); ax.set_ylabel('MNC'); ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def curvas(models,path):
    fig,axes = plt.subplots(2,2,figsize=(14,10))
    for name,record in models.items():
        h = np.load(record['exp']/'historial.npy',allow_pickle=False)
        for ax,col,title in zip(axes.ravel(),[1,3,2,4],['Train loss','Val loss','Train accuracy','Val accuracy']):
            ax.plot(h[:,0],h[:,col],label=record['nombre']); ax.set_title(title)
            ax.set_xlabel('Epoca'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Curvas de entrenamientos: VAL multicampana y VAL 2019/2020 son conjuntos distintos',fontsize=11)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
