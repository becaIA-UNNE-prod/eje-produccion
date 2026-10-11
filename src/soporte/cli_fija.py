"""Validación temprana para entradas que conservan configuración fija.

Solo usa la biblioteca estándar: se llama antes de importar código operativo.
"""
import argparse

DESCRIPCIONES = {
    'recuperar_multicampana': 'Audita y reconstruye composites corruptos; reanuda preparación sin entrenamiento.',
    'moldear_mascaras': 'Genera la máscara de prueba original fija sobre JP2.',
    'composites_mensuales': 'Genera composites JP2 con la mediana mensual original.',
    'composites_2019_2020': 'Genera composites JP2 2019/2020 con la primera fecha válida.',
    'generar_arrays_originales': 'Genera arrays con el calendario y remapeo originales 0–8.',
}


def validar_entrada_fija(nombre, argv=None):
    """Ayuda sale con 0; argumentos no admitidos con 2, antes de operar."""
    parser = argparse.ArgumentParser(
        description=DESCRIPCIONES[nombre],
        epilog=('Sin argumentos ejecuta la configuración fija existente y puede '
                'modificar datos y salidas. Solo se admiten -h/--help; '
                'esta entrada no configura rutas ni parámetros científicos.'),
        allow_abbrev=False,
    )
    return parser.parse_args(argv)
