#!/bin/bash
cd ..

CUDA_VISIBLE_DEVICES=1,0 python launch.py --nproc_per_node=2 trainval.py \
    --num_workers 2 \
    --device 'cuda' \
    --train_batch 8 \
    --train_size 680 680 \
    --train_data '/home/1621437/work/data/cityscapes' \
    --max_epochs 350 \
	--learning_rate 1e-3 \
	--save_interval 10 \
	--report_interval 10 \
	--momentum 0.9 \
	--weight_decay 5e-4 \
	--max_iters 64750 \
    --power 0.9 \
	--scale_limits 0.5 2.0 \
	--scale_step 0.25 \
	--val_batch 2 \
    --val_size 2048 1024 \
    --val_data '/home/1621437/work/data/cityscapes' \
    --poly_lr \
    --sync_bn
