#!/bin/bash
#SBATCH --account=aip-gidelgau
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0                    # “alloc as needed” on Alliance
#SBATCH --job-name=sweep_g2

module load python/3.10.13
module load httpproxy
source $SCRATCH/keller/bin/activate

# Set wandb API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946

for batch_per_device in 2 16; do
    for lr_multiplier in .1; do
        for cooldown_frac in 0.8; do
            for warmup_frac in 0.02; do
                for weight_decay in 1e-3; do
                    echo "Running with lr_multiplier=$lr_multiplier, cooldown_frac=$cooldown_frac, warmup_frac=$warmup_frac, weight_decay=$weight_decay, batch_per_device=$batch_per_device"
                    torchrun --standalone --nproc_per_node=4 train_gpt_modded_torch_simpl.py \
                        -lr_multiplier $lr_multiplier \
                        -cooldown_frac $cooldown_frac \
                        -warmup_frac $warmup_frac \
                        -weight_decay $weight_decay \
                        -batch_per_device $batch_per_device
                done
            done
        done
    done
done