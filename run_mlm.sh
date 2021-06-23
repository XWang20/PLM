#!/bin/sh
  
python -m torch.distributed.launch --nproc_per_node=6 ./mlm_emoji.py \
             --model_name_or_path ../TwitterPLM/checkpoint-17400 \
             --output_dir /data/private/wangxing/TwitterPLM/TwitterPLM/emoji \
             --seed 42 \
             --do_train \
             --train_file /data/private/wangxing/TwitterPLM/sentences/emoji/normalization/emoji.txt \
             --preprocessing_num_workers 32 \
             --line_by_line True \
             --max_seq_length 512 \
             --num_train_epochs 1 \
             --per_device_train_batch_size 6 \
             --per_device_eval_batch_size 6 \
             --gradient_accumulation_steps 42 \
             --ddp_find_unused_parameters False \
             --save_steps 1000 \
             --cache_dir ./.cache \
             --fp16 \
             --learning_rate 7e-4 \
             --weight_decay 0.01 \
             --adam_beta1 0.9 \
             --adam_beta2 0.98 \
             --adam_epsilon 1e-6 \
             --lr_scheduler_type linear \
             --logging_steps 500 \
             --dataloader_num_workers 32
