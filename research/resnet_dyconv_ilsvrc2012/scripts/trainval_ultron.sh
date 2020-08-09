#!/bin/bash
cd ..

python launch.py --nproc_per_node=2 trainval.py \
    --device 'cuda' \
    --train_batch 64 \
    --train_size 224 \
    --train_data '/home/ljy/work/data/ilsvrc2012/Data' \
    --max_epochs 250 \
	--learning_rate 0.1 \
	--save_interval 1 \
	--report_interval 10 \
	--momentum 0.9 \
	--weight_decay 4e-5 \
    --step_size 30 \
    --gamma 0.1 \
	--val_batch 128 \
    --val_resize 256 \
    --val_size 224 \
    --val_data '/home/ljy/work/data/ilsvrc2012/Data'
