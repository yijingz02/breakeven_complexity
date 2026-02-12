#!/bin/bash

export HOME=$PWD
echo $HOME/

nvidia-smi

sh Anaconda3-2024.10-1-Linux-x86_64.sh -b -p $HOME/anaconda3
export PATH=$HOME/anaconda3/bin:$PATH
source $HOME/anaconda3/etc/profile.d/conda.sh
hash -r

conda config --set always_yes yes --set changeps1 no
conda create -n pde python=3.10
conda activate pde
echo "activated"

cd poseidon
pip install -r requirements.txt
pip install -e .
pip install -U wandb
pip install "numpy<2.0"

pip install scipy

ls

echo "<-------->"

DATA_PATH="data"
SAVE_PATH="."

BUDGET=1e3
ndatas=(500 1000 1400 2000 2800 4000 5600 8000 16000 20000)
for ndata in "${ndatas[@]}"; do
    accelerate launch scOT/train.py \
        --config configs/kuramoto_sivashinsky.yaml \
        --wandb_project_name poseidon_KS \
        --checkpoint_path $SAVE_PATH \
        --data_path $DATA_PATH \
        --ndata=$ndata \
        --time_budget=$BUDGET \
        --batch_size=300 \
        --lr=1e-3
done