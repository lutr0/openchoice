#!/usr/bin/env bash
# Build locally; no precompiled libraries or model weights are downloaded here.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-cpu}
RUNTIME=${2:-"$PROJECT/.runtime"}
PIN=4364bf7232e65c34eca8d9500c5464389662de6b
[[ "$MODE" == cpu || "$MODE" == cuda ]] || { echo 'Usage: build_llamacpp.sh [cpu|cuda] [runtime-dir]' >&2; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'Only Linux builds have been verified.' >&2; exit 2; }
mkdir -p "$RUNTIME"
RUNTIME=$(cd "$RUNTIME" && pwd)
SOURCE="$RUNTIME/llama.cpp"
if [[ ! -d "$SOURCE/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$SOURCE"
    git -C "$SOURCE" checkout --detach "$PIN"
fi
[[ $(git -C "$SOURCE" rev-parse HEAD) == "$PIN" ]] || { echo 'Unexpected llama.cpp revision' >&2; exit 1; }
[[ -z $(git -C "$SOURCE" status --porcelain --untracked-files=no) ]] || { echo 'llama.cpp source has local modifications' >&2; exit 1; }
NATIVE=ON
DISPATCH=OFF
if [[ "${PORTABLE:-0}" == 1 ]]; then
    NATIVE=OFF
    DISPATCH=ON
fi
FLAGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE="$NATIVE" -DGGML_BLAS=OFF
       -DGGML_BACKEND_DL="$DISPATCH" -DGGML_CPU_ALL_VARIANTS="$DISPATCH"
       -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF
       -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF)
# uv supplies pinned CMake even on a host that has only git, make, and a C++ compiler.
CMAKE=(uv run --frozen --project "$PROJECT" --group build cmake)
for KIND in cpu $([[ "$MODE" == cuda ]] && echo cuda || true); do
    CUDA=OFF
    [[ "$KIND" == cuda ]] && CUDA=ON
    BACKEND_DIR=
    [[ "$DISPATCH" == ON ]] && BACKEND_DIR="$SOURCE/build-$KIND/bin"
    "${CMAKE[@]}" -S "$SOURCE" -B "$SOURCE/build-$KIND" "${FLAGS[@]}" \
        -DGGML_CUDA="$CUDA" -DGGML_BACKEND_DIR="$BACKEND_DIR"
    "${CMAKE[@]}" --build "$SOURCE/build-$KIND" --target llama mtmd llama-quantize --parallel "${BUILD_JOBS:-4}"
    "${CXX:-c++}" -O3 -std=c++17 -fPIC -shared "$PROJECT/openchoice/llamacpp_bridge.cpp" \
        -I"$SOURCE/include" -I"$SOURCE/ggml/include" -I"$SOURCE/tools/mtmd" \
        -L"$SOURCE/build-$KIND/bin" -Wl,-rpath,"\$ORIGIN/llama.cpp/build-$KIND/bin" \
        -lllama -lmtmd -lggml -lggml-base -o "$RUNTIME/llamacpp-bridge-$KIND.so.tmp"
    mv "$RUNTIME/llamacpp-bridge-$KIND.so.tmp" "$RUNTIME/llamacpp-bridge-$KIND.so"
done
uv run --project "$PROJECT" python - "$RUNTIME" "$PIN" "$PROJECT/openchoice/llamacpp_bridge.cpp" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, commit, source = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
(root / 'build.json').write_text(json.dumps({
    'llama_commit': commit,
    'bridge_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
}, indent=2) + '\n')
print(f'Built {root}; serve with python -m openchoice.server --model ./OpenChoice-9B-GGUF')
PY
