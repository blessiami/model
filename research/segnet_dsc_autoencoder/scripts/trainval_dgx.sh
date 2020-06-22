#!/bin/bash
cd ..

python trainval.py \
    --num_workers 7 \
    --device 'cuda' \
    --train_batch 14 \
    --train_size 769 769 \
    --train_data '/mfc/data/mobis/synthetic/temp/public/Cityscapes/' \
    --base_weight 'xception_imagenet.pth' \
    --max_epochs 200 \
	--learning_rate 7e-3 \
	--save_interval 10 \
	--report_interval 10 \
	--momentum 0.9 \
	--weight_decay 4e-5 \
	--power 0.9 \
	--max_iters 40000 \
	--scale_limits 0.5 2.0 \
	--scale_step 0.25 \
	--val_batch 42 \
    --val_size 2049 1025 \
    --val_data '/mfc/data/mobis/synthetic/temp/public/Cityscapes/'
