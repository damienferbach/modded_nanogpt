for batch_per_device in 16; do
    for lr_multiplier_1 in 0.06; do
        for lr_multiplier_2 in .03; do
            for cooldown_frac in 0.8; do
                for warmup_frac in 0.02; do
                    for weight_decay in 1e-3; do
                        echo "Running with lr_multiplier=$lr_multiplier, cooldown_frac=$cooldown_frac, warmup_frac=$warmup_frac, weight_decay=$weight_decay, batch_per_device=$batch_per_device"
                        torchrun --standalone --nproc_per_node=1 train_gpt_modded_tanea.py \
                            -lr_multiplier_1 $lr_multiplier_1 \
                            -lr_multiplier_2 $lr_multiplier_2 \
                            -cooldown_frac $cooldown_frac \
                            -warmup_frac $warmup_frac \
                            -weight_decay $weight_decay \
                            -batch_per_device $batch_per_device
                    done
                done
            done
        done
    done
done