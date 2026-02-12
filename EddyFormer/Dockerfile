FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
  
RUN apt-get update && apt-get install -y python3 python3-pip libopenmpi-dev openmpi-bin git build-essential cmake
RUN pip install --no-cache-dir numpy scipy cvxpy ilupp pyamg mosek
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/ASK-Berkeley/EddyFormer.git
WORKDIR /app/EddyFormer

RUN pip install --upgrade "jax[cuda12]>=0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install -e .

RUN pip install jupyter matplotlib numpy scipy

RUN pip install huggingface-hub

RUN pip install wandb

WORKDIR /workspace
