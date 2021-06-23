#!/bin/sh

python -m torch.distributed.launch --nproc_per_node=4 ./run_nsp.py \
             --model_name_or_path roberta-base \
             --output_dir ./cashed2 \
             --seed 42 \
             --do_train \
             --num_train_epochs 20 \
             --per_device_train_batch_size 6 \
             --per_device_eval_batch_size 6 \
             --gradient_accumulation_steps 42 \
             --ddp_find_unused_parameters False \
             --save_steps 500 \
             --cache_dir ./cache \
             --fp16 \
             --learning_rate 7e-4 \
             --weight_decay 0.01 \
             --adam_beta1 0.9 \
             --adam_beta2 0.98 \
             --adam_epsilon 1e-6 \
             --lr_scheduler_type linear \
             --logging_steps 100 \
             --dataloader_num_workers 2
