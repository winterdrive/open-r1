# 進入專案目錄
cd /mnt/openr1_data_disk/OpenR1_PT/open-r1
mkdir -p data/gemma-3-1b-it
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
pip install --upgrade transformers
pip install git+https://github.com/huggingface/transformers.git

# Huggingface 與 wandb 登入（注意：以下指令使用標準輸入方式傳入 API key）
huggingface-cli login
wandb login

# 開始訓練前，先清乾淨畫面
clear

# 執行 accelerate 指令開始訓練
accelerate launch   --config_file recipes/accelerate_configs/8v100.yaml   --num_processes 8   --main_process_port 29500   --machine_rank 0   --mixed_precision fp16   src/open_r1/sft.py   --model_name_or_path google/gemma-3-1b-it   --dataset_name open-r1/OpenR1-Math-220k   --learning_rate 0.00001   --num_train_epochs 5   --max_seq_length 1024   --per_device_train_batch_size 4   --gradient_accumulation_steps 2   --gradient_checkpointing   --fp16   --output_dir data/gemma-3-1b-it   --log_level info   --dataloader_num_workers 4 \


# GPU 使用率監控:
watch -n 1 nvidia-smi

# benchmark
#!/bin/bash
# 使用 8 個 GPU 進行資料並行評估，並使用本地模型
NUM_GPUS=8
MODEL="/mnt/openr1_data_disk/OpenR1_PT/open-r1/data/gemma-3-1b-it"
# 由於 Tesla V100 不支援 bfloat16，這裡使用 half (float16)
MODEL_ARGS="pretrained=$MODEL,dtype=half,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={\"max_new_tokens\":32768,\"temperature\":0.6,\"top_p\":0.95}"
TASK="aime24"
OUTPUT_DIR="data/evals/$(basename $MODEL)"

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

