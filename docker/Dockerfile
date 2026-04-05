FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

ARG DOCKERCOMPOSE_UID=1001
ARG DOCKERCOMPOSE_GID=1001

ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKERCOMPOSE_UID=${DOCKERCOMPOSE_UID}
ENV DOCKERCOMPOSE_GID=${DOCKERCOMPOSE_GID}
ENV DOCKERCOMPOSE_USER=developer
ENV DOCKERCOMPOSE_GROUP=developer

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang-format \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    zsh \
 && groupadd --gid ${DOCKERCOMPOSE_GID} ${DOCKERCOMPOSE_GROUP} \
 && useradd --uid ${DOCKERCOMPOSE_UID} --gid ${DOCKERCOMPOSE_GID} --create-home --shell /usr/bin/zsh ${DOCKERCOMPOSE_USER} \
 && mkdir -p /workspace \
 && chown -R ${DOCKERCOMPOSE_UID}:${DOCKERCOMPOSE_GID} /workspace /home/${DOCKERCOMPOSE_USER} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

USER ${DOCKERCOMPOSE_USER}

CMD ["sleep", "infinity"]
