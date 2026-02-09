#!/bin/bash
# shell script totalmente vibe-codeado para hacer links simbólicos

# --- CONFIGURACIÓN ---
# La ruta completa donde están las carpetas originales (lo que mostraste en el pwd)
SOURCE_DIR="/mnt/yacy_3/prod/dat/S4A/2019"

# La ruta donde querés que aparezcan los links
TARGET_DIR="/home1/ferreyra/S4A-Models/coco_files"
# ---------------------

##mkdir -p "$TARGET_DIR"

echo "Creando links simbólicos..."
echo "De: $SOURCE_DIR"
echo "A:  $TARGET_DIR"
echo "-------------------"

# 2. Iterar sobre cada carpeta dentro del source
for full_path in "$SOURCE_DIR"/*; do
    if [ -d "$full_path" ]; then
        # Extraer solo el nombre de la carpeta (ej: 31TBF)
        dirname=$(basename "$full_path")
        
        # Crear el link simbólico
        # Sintaxis: ln -s TARGET LINK_NAME
        ln -s "$full_path" "$TARGET_DIR/$dirname"
        
        echo "Link creado: $dirname"
    fi
done

echo "-------------------"
echo "Listo."
