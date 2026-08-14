# Leyenda del Mapa Nacional de Cultivos (MNC) — INTA

## Advertencia importante

La leyenda numerica del MNC **no esta embebida en el GeoTIFF** (no tiene colormap
ni tags). Hay que sacarla del informe PDF que acompana al dataset en Zenodo.

Una leyenda incorrecta obtenida de fuentes secundarias nos costo un experimento
completo: teniamos **soja y maiz invertidos**, lo que produjo 46% de Test Accuracy
en un modelo que en realidad funcionaba bien.

## Leyenda oficial — Verano 2020 (Tabla 2 del informe)

| IDVER | Clase          |
|-------|----------------|
| 10    | Maiz           |
| 11    | Soja           |
| 12    | Girasol        |
| 13    | Poroto         |
| 14    | Cana de azucar |
| 15    | Algodon        |
| 16    | Mani           |
| 17    | Arroz          |
| 18    | Sorgo          |
| 19    | Girasol-CV     |
| 21    | Barbecho       |
| 22    | No agricola    |
| 255   | Sin datos      |

Fuente: "Mapa Nacional de Cultivos campana 2019_2020.pdf", Zenodo
https://zenodo.org/records/8286310

## Cobertura por zona en Cordoba

Cordoba corresponde a las zonas PAS III (centro-norte) y IV (sur).
Segun la Tabla 4 del informe:

- **Maiz y Soja**: mapeados en ambas zonas.
- **Mani**: solo en zona IV (sur). Ausente en los tiles del este.
- **Sorgo**: solo en zona III, y el informe indica que las clases poco
  representadas (incluido sorgo) tuvieron exactitudes bajas.

Por eso el esquema de entrenamiento usa **5 clases**:
`0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani`

## Exactitud reportada por el INTA

Mapas de verano 2020: exactitud general 0,88 e indice Kappa 0,82.
Esto acota el techo alcanzable por cualquier modelo entrenado contra
estas etiquetas.
