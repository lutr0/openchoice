# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.26 AS uv
FROM python:3.12-slim-trixie AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" HF_HOME=/tmp/huggingface
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv

FROM base AS dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-cache

FROM dependencies AS build
ARG BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/build_llamacpp.sh scripts/build_llamacpp.sh
COPY openchoice/llamacpp_bridge.cpp openchoice/llamacpp_bridge.cpp
# Compile portable backends, including a baseline, and choose CPU features at runtime.
RUN PORTABLE=1 BUILD_JOBS=${BUILD_JOBS} nice -n 10 bash scripts/build_llamacpp.sh cpu \
    && mkdir -p /runtime/llama.cpp/build-cpu/bin \
    && cp -a .runtime/llama.cpp/build-cpu/bin/*.so* /runtime/llama.cpp/build-cpu/bin/ \
    && cp .runtime/llamacpp-bridge-cpu.so .runtime/build.json /runtime/ \
    && cp .runtime/llama.cpp/LICENSE /runtime/LICENSE-llama.cpp

FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 openchoice \
    && useradd --uid 10001 --gid openchoice --create-home openchoice \
    && mkdir /models && chown openchoice:openchoice /models
COPY --from=dependencies /app/.venv /app/.venv
COPY --from=build /runtime /app/.runtime
COPY openchoice/__init__.py openchoice/server.py openchoice/llamacpp.py openchoice/native.py openchoice/inference.py openchoice/integrity.py openchoice/llamacpp_bridge.cpp /app/openchoice/
COPY LICENSE NOTICE /app/
COPY --chmod=755 scripts/container-entrypoint.sh /usr/local/bin/openchoice
LABEL org.opencontainers.image.source="https://github.com/lutr0/openchoice" \
      org.opencontainers.image.description="OpenChoice-9B CPU decision server (model weights mounted separately)" \
      org.opencontainers.image.licenses="Apache-2.0"
USER 10001:10001
EXPOSE 8768
HEALTHCHECK --interval=10s --timeout=3s --start-period=5m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8768/health', timeout=2).read()"]
ENTRYPOINT ["openchoice"]
CMD []
