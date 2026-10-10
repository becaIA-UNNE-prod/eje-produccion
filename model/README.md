# Arquitecturas de segmentación

Las implementaciones de los cuatro modelos están en esta carpeta.
Importación pública: from model import SimpleUNet, TransUNet, UTAE, FCRTransUNet.

| Archivo | Contenido y estructura |
|---|---|
| unet.py | SimpleUNet: encoder 64→128→256→512, cuello 1024, decoder con cuatro conexiones de salto. |
| transunet.py | TransUNet: misma estructura espacial, con Transformer en el cuello de botella. |
| utae.py | UTAE: encoder compartido por mes, atención temporal LTAE y agregación temporal de las conexiones de salto. Incluye bloques temporales propios. |
| fcrtransunet.py | FCRTransUNet: Transformer, FEM en encoder, Coordinate Attention en saltos y Reverse Attention en decoder. Incluye bloques específicos de esta adaptación. |
| bloques.py | DoubleConv y TransformerBottleneck, compartidos entre arquitecturas. |
| __init__.py | API pública de importación. |

Entrada: tensor (batch, canales, alto, ancho); salida: logits
(batch, clases, alto, ancho). El flujo vigente usa 24 canales
(seis meses × cuatro bandas) y cinco clases. Para UTAE usar explícitamente
UTAE(24, 5, T=6); se conserva el default histórico T=7.
Los canales deben conservar el orden mensual y B02/B03/B04/B08.
Las dimensiones espaciales deben ser compatibles con cuatro reducciones por dos.

Esta separación conserva los cuerpos de las clases, atributos, parámetros y
claves de state_dict; no cambia las arquitecturas ni el protocolo científico.
utils/model.py reexporta las clases y auxiliares como compatibilidad con imports
anteriores. Los docstrings conservan contexto histórico; las menciones a siete
meses no definen el protocolo vigente.

Esta carpeta contiene código. Los pesos y checkpoints siguen en el
almacenamiento externo definido por config/almacenamiento.json.
