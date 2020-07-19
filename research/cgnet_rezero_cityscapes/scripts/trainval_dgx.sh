#!/bin/bash
cd ..

python launch.py --nproc_per_node=2 trainval.py \
    --num_workers 2 \
    --device 'cuda' \
    --train_batch 6 \
    --train_size 769 769 \
    --train_data '/mfc/project/miner/public/cityscapes' \
    --base_weight '/mfc/project/miner/weight/basenets/espnetv2_s_1.5.pth' \
    --max_epochs 490 \
    --learning_rate 0.01 \
    --save_interval 10 \
    --report_interval 10 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --power 0.9 \
	--max_iters 120000 \
    --scale_limits 0.5 2.0 \
    --scale_step 0.25 \
    --val_batch 6 \
    --val_size 2049 1025 \
    --val_data '/mfc/project/miner/public/cityscapes' \
    --sync_bn \
    --poly_lr \
    --init_method 'tcp://10.230.102.46:6053'
