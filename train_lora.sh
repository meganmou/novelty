# Adapted from FastChat train_lora.sh

deepspeed finetuning/fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path finetuning/individual_chapter_training_data.json \
    --output_dir checkpoints/single-chapter-finetuned-vicuna \
    --num_train_epochs 10 \
    --fp16 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --eval_steps 100  \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2500 \
    --q_lora True \
    --deepspeed ./scripts/zero2.json \
    --gradient_checkpointing True \
    --flash_attn False \
    --cache_dir "/path/to/cache"
