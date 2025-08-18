#! /bin/bash
#SBATCH --output=modded-nanogpt-%j.out
#SBATCH --error=modded-nanogpt-%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=64

module load python/3.10
source $SCRATCH/keller/bin/activate

# Set wandb API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY=bece9f2099e3e85e0ae9922002616cf20bd26946

for batch_per_device in 2 16; do
    for lr_multiplier in .1; do
        for cooldown_frac in 0.8; do
            for warmup_frac in 0.02; do
                for weight_decay in 1e-3; do
                    echo "Running with lr_multiplier=$lr_multiplier, cooldown_frac=$cooldown_frac, warmup_frac=$warmup_frac, weight_decay=$weight_decay, batch_per_device=$batch_per_device"
                    torchrun --standalone --nproc_per_node=4 train_gpt_modded.py \
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