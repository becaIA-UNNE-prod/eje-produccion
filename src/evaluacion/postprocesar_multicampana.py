"""Despacha al TEST vigente o al protocolo anterior pedido con --historico."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--historico' in args:
        args.remove('--historico')
        if '--lock-adquirido' in args:
            args.remove('--lock-adquirido')
        from src.historico.postprocesar_multicampana import main as ejecutar
    else:
        from src.evaluacion.test_actual import main as ejecutar
    return ejecutar(args)


if __name__ == '__main__':
    main()
