#!/bin/bash
cd ..

python trainval.py \
    --resume 'default' \
    --num_workers 2 \
    --device 'cuda' \
    --train_batch 8 \
    --train_size 480 360 \
    --train_data '/home/ubuntu/work/data/camvid' \
    --max_epochs 5000 \
    --learning_rate 1 \
    --save_interval 10 \
    --report_interval 10 \
    --momentum 0.9 \
    --weight_decay 4e-5 \
    --step_size 100 \
    --gamma 0.94 \
    --scale_limits 0.5 2.0 \
    --scale_step 0.25 \
    --val_batch 8 \
    --val_size 480 360 \
    --val_data '/home/ubuntu/work/data/camvid'
