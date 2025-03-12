### Open-R1 PT 指令紀錄

1. server.sh: 用來為 server 安裝環境
2. train_1.sh: 用來單片 V100 跑的指令
3. train_8.sh: 用來 8 片 V100 跑的指令

``` bash 1片V100直接跑的指令
accelerate launch \
  --config_file recipes/accelerate_configs/zero_v100.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  --machine_rank 0 \
  --mixed_precision fp16 \
  src/open_r1/sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name open-r1/OpenR1-Math-220k \
  --learning_rate 0.0008 \
  --num_train_epochs 5 \
  --packing \
  --max_seq_length 8192 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --fp16 \
  --output_dir data/Qwen2.5-1.5B-Open-R1-Distill \
  --log_level info \
  --dataloader_num_workers 4 \ 
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{
    "backward_prefetch": "backward_pre",
    "forward_prefetch": true,
    "limit_all_gathers": true,
    "use_orig_params": true
  }'
