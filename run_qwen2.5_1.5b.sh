# 進入專案目錄
cd /mnt/openr1_data_disk/OpenR1_PT/open-r1
mkdir -p data/Qwen2.5-1.5B-Open-R1-Distill-0317
mkdir -p data_cache/deepseed_cache
mkdir -p data_cache/huggingface_cache
mkdir -p data_cache/huggingface_home

# 設定環境變數
export TORCH_EXTENSIONS_DIR=/mnt/openr1_data_disk/torch_extensions
export DEEPSPEED_AUTOTUNE_CACHE_DIR=/mnt/openr1_data_disk/data_cache/deepseed_cache
export HUGGINGFACE_HUB_CACHE=/mnt/openr1_data_disk/data_cache/huggingface_cache
export HF_HOME=/mnt/openr1_data_disk/data_cache/huggingface_home


# 啟用 Python 虛擬環境
source TWM_R1/bin/activate

# Huggingface 與 wandb 登入（注意：以下指令使用標準輸入方式傳入 API key）
huggingface-cli login
wandb login

# 執行 accelerate 指令開始訓練
accelerate launch   --config_file recipes/accelerate_configs/8v100.yaml   --num_processes 8   --main_process_port 29500   --machine_rank 0   --mixed_precision fp16   src/open_r1/sft.py   --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct   --dataset_name open-r1/OpenR1-Math-220k   --learning_rate 0.00005   --num_train_epochs 1   --max_seq_length 2048   --per_device_train_batch_size 6   --gradient_accumulation_steps 2   --gradient_checkpointing   --fp16   --output_dir data/Qwen2.5-1.5B-Open-R1-Distill-0317   --log_level info   --dataloader_num_workers 4 \

# GPU 使用率監控:
watch -n 1 nvidia-smi