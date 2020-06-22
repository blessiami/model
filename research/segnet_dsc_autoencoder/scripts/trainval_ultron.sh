#!/bin/bash
cd ..

python launch.py --nproc_per_node=2 trainval.py \
    --num_workers 2 \
    --device 'cuda' \
    --train_batch 8 \
    --train_size 768 768 \
    --train_data '/home/ljy/Work/Data/Cityscapes' \
    --max_epochs 1000 \
	--learning_rate 7e-3 \
	--save_interval 10 \
	--report_interval 10 \
	--momentum 0.9 \
	--weight_decay 2e-4 \
	--step_size 20 \
    --gamma 0.94 \
	--scale_limits 0.5 2.0 \
	--scale_step 0.25 \
	--val_batch 3 \
    --val_size 2048 1024 \
    --val_data '/home/ljy/Work/Data/Cityscapes' \
    --sync_bn
