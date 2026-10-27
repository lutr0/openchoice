#!/bin/sh
set -eu
if [ "${1:-}" = download ]; then
    shift
    if [ "$#" -eq 0 ]; then
        echo 'Usage: docker run ... openchoice:cpu download lutr0/OpenChoice-9B' >&2
        exit 2
    fi
    exec hf download "$@" --local-dir /models
fi
exec nice -n 10 python -m openchoice.server --model /models --host 0.0.0.0 --port 8768 "$@"
