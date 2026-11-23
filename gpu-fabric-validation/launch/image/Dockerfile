# nccl-tests image used by every launcher. Not built in CI (no GPU runners); the tag is
# what the manifests reference. Pin the CUDA base and nccl-tests tag together.
FROM nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG NCCL_TESTS_TAG=v2.13.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        git openmpi-bin libopenmpi-dev openssh-server openssh-client \
        rdma-core ibverbs-utils infiniband-diags perftest ethtool \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${NCCL_TESTS_TAG} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && make -C /opt/nccl-tests MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi -j"$(nproc)"
ENV PATH=/opt/nccl-tests/build:${PATH}
