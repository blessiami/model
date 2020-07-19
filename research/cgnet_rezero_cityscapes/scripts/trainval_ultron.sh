#!/bin/bash
cd ..

python trainval.py \
    --num_workers 2 \
    --device 'cuda' \
    --train_batch 28 \
    --train_size 680 680 \
    --train_data '/home/ljy/work/data/cityscapes' \
    --max_epochs 350 \
	--learning_rate 1e-3 \
	--save_interval 10 \
	--report_interval 10 \
	--momentum 0.9 \
	--weight_decay 5e-4 \
	--step_size 20 \
    --gamma 0.94 \
	--scale_limits 0.5 2.0 \
	--scale_step 0.25 \
	--val_batch 28 \
    --val_size 2048 1024 \
    --val_data '/home/ljy/work/data/cityscapes' \
    --sync_bn

