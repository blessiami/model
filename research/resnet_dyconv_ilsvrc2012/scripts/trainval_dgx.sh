#!/bin/bash
cd ..

python launch.py --nproc_per_node=8 trainval.py \
    --device 'cuda' \
    --train_batch 128 \
    --train_size 224 \
    --train_data '/mfc/project/miner/public/ilsvrc2012/Data' \
    --max_epochs 1000 \
    --learning_rate 0.4 \
    --save_interval 1 \
    --report_interval 10 \
    --momentum 0.9 \
    --weight_decay 4e-5 \
    --step_size 180 \
    --gamma 0.1 \
    --val_batch 256 \
    --val_resize 256 \
    --val_size 224 \
    --val_data '/mfc/project/miner/public/ilsvrc2012/Data'
