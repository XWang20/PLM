#!/bin/sh
  
python -m torch.distributed.launch --nproc_per_node=4 ./mlm_emoji.py \
             --model_name_or_path ../TwitterPLM/checkpoint-17400 \
             --output_dir /data/private/wangxing/TwitterPLM/TwitterPLM/emoji_test \
             --seed 320 \
             --do_train \
             --train_file /data/private/wangxing/TwitterPLM/sentences/emoji/normalization/0.txt \
             --preprocessing_num_workers 8 \
             --line_by_line True \
             --max_seq_length 512 \
             --num_train_epochs 1 \
             --per_device_train_batch_size 6 \
             --per_device_eval_batch_size 6 \
             --gradient_accumulation_steps 42 \
             --ddp_find_unused_parameters False \
             --save_steps 100 \
             --cache_dir ./.cache2 \
             --fp16 \
             --learning_rate 1e-3 \
             --weight_decay 0.01 \
             --adam_beta1 0.9 \
             --adam_beta2 0.98 \
             --adam_epsilon 1e-6 \
             --lr_scheduler_type linear \
             --logging_steps 100 \
             --dataloader_num_workers 8