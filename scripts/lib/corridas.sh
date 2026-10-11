declare -A CORDOBA_LOCK_FDS
# Se carga desde la raíz del repo. No modifica destinos de datos/resultados.
cordoba_run_init() {
    local etapa=$1 nombre=$2
    shift 2
    if [[ -z ${CORDOBA_RUN_DIR:-} ]]; then
        exec "${CORDOBA_PY_REGISTRO:-${PY:-$HOME/s4a-venv/bin/python}}" scripts/ejecutar_con_log.py \
            --etapa "$etapa" --nombre "$nombre" -- bash "$0" "$@"
    fi
    CORDOBA_RUN_DIR=$(realpath -m -- "$CORDOBA_RUN_DIR")
    if [[ "$CORDOBA_RUN_DIR" == "$PWD" || "$CORDOBA_RUN_DIR" == "$PWD/"* ]]; then
        echo 'CORDOBA_RUN_DIR debe estar fuera del repositorio' >&2; return 2
    fi
    export CORDOBA_RUN_DIR
    export CORDOBA_LOG_DIR="$CORDOBA_RUN_DIR/logs"
    export CORDOBA_INVENTARIOS_DIR="$CORDOBA_RUN_DIR/inventarios"
    export CORDOBA_RUNTIME_ROOT="${CORDOBA_RUNTIME_ROOT:-/mnt/yacy_3/prod_bis/ferreyra/cordoba_ia_unne/runtime}"
    mkdir -p "$CORDOBA_LOG_DIR/workers" "$CORDOBA_INVENTARIOS_DIR" "$CORDOBA_RUNTIME_ROOT/locks"
}

cordoba_registrar() {
    [[ -f "$CORDOBA_RUN_DIR/run.json" ]] || return 0
    "${CORDOBA_PY_REGISTRO:-${PY:-$HOME/s4a-venv/bin/python}}" scripts/registrar_rutas_corrida.py \
        --run "$CORDOBA_RUN_DIR" "$@"
}

cordoba_lock() {
    # Adquirir después del lock histórico: mismo orden en todos los consumidores.
    local nombre=$1 descriptor
    exec {descriptor}>"$CORDOBA_RUNTIME_ROOT/locks/$nombre.lock"
    if [[ ${2:-} == esperar ]]; then
        flock "$descriptor"
    else
        flock -n "$descriptor" || { echo "Ya hay una ejecución activa: $nombre"; return 1; }
    fi
    CORDOBA_LOCK_FDS[$nombre]=$descriptor
    # El descriptor permanece abierto hasta finalizar el shell.
}

cordoba_unlock() {
    local descriptor=${CORDOBA_LOCK_FDS[$1]}
    flock -u "$descriptor"
    exec {descriptor}>&-
    unset 'CORDOBA_LOCK_FDS[$1]'
}

cordoba_vincular() {
    [[ -f "$CORDOBA_RUN_DIR/run.json" ]] || return 0
    "${CORDOBA_PY_REGISTRO:-${PY:-$HOME/s4a-venv/bin/python}}" scripts/registrar_rutas_corrida.py         --run "$CORDOBA_RUN_DIR" --vinculo "$1"
}
