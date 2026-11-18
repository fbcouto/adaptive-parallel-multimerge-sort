#!/bin/sh
# Gera a copia do motor Rust para um projeto derivado.
#
#   sh scripts/copia_do_motor.sh <destino>               sem o modulo de testes
#   sh scripts/copia_do_motor.sh --com-testes <destino>  com os testes (o projeto
#                                                        precisa de `rand` em
#                                                        dev-dependencies)
#
# Os derivados NAO dependem deste crate por caminho: o build.rs daqui compila
# C++ com OpenMP, e isso nao pode entrar no build de uma extensao do Postgres
# ou de um modulo Python. Cada um leva uma copia de src/multimerge.rs, que so
# depende de rayon e da std.
set -e
com_testes=0
if [ "$1" = "--com-testes" ]; then com_testes=1; shift; fi
[ -n "$1" ] || { echo "uso: $0 [--com-testes] <arquivo de destino>" >&2; exit 1; }
orig="$(dirname "$0")/../src/multimerge.rs"

{
  if [ "$com_testes" = 1 ]; then
    printf '// COPIA de adaptive-parallel-multimerge-sort/src/multimerge.rs, com os testes.\n'
    printf '// O original e a fonte da verdade: nao edite esta copia -- altere la, rode os\n'
    printf '// testes la e copie de novo.\n\n'
    tr -d '\r' < "$orig"
  else
    printf '// COPIA de adaptive-parallel-multimerge-sort/src/multimerge.rs.\n'
    printf '// O original e a fonte da verdade e guarda os testes do motor; nao edite\n'
    printf '// esta copia -- altere la, rode os testes la e copie de novo.\n\n'
    # Tudo ate a linha de separacao que antecede o bloco "//  TESTES".
    n=$(tr -d '\r' < "$orig" | grep -n '^//  TESTES$' | cut -d: -f1)
    [ -n "$n" ] || { echo "marcador de testes nao encontrado em $orig" >&2; exit 1; }
    tr -d '\r' < "$orig" | head -n $((n - 3))
  fi
} > "$1"
echo "ok: $1 ($(wc -l < "$1") linhas)"
