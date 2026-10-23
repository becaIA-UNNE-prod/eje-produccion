Nuevo TEST: 20HNJ y 20HNH, diciembre-mayo en cuatro campanas.
U-Net multicampana congelada; sin reentrenar ni seleccionar pesos con el TEST nuevo.
Se genera referencia completa y referencia que excluye las grillas completas de TRAIN y VAL, union de las tres campanas de entrenamiento.
2021/2022 agrega generalizacion temporal; las otras campanas son nuevas zonas en anos ya entrenados.
Referencia INTA, no validacion independiente de campo. Todos los compuestos nuevos usan SCL; 2019/2020 original tuvo otro preprocesamiento.
El split de esta evaluacion se guarda separado; no se cambia la corrida original.
No se generan arrays X/Y innecesarios para inferencia de mapas completos.
