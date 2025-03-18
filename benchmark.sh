# 進入專案目錄
cd /mnt/openr1_data_disk/OpenR1_PT/open-r1

# 啟用 Python 虛擬環境
source TWM_R1/bin/activate

# Huggingface 與 wandb 登入（注意：以下指令使用標準輸入方式傳入 API key）
huggingface-cli login
wandb login

# 設定環境變數
export TORCH_EXTENSIONS_DIR=/mnt/openr1_data_disk/torch_extensions
export DEEPSPEED_AUTOTUNE_CACHE_DIR=/mnt/openr1_data_disk/data_cache/deepseed_cache
export HUGGINGFACE_HUB_CACHE=/mnt/openr1_data_disk/data_cache/huggingface_cache
export HF_HOME=/mnt/openr1_data_disk/data_cache/huggingface_home

#Gemma 3

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


#Qwen2.5-1.5B-Open-R1-Distill-0317

#!/bin/bash
# 使用 8 個 GPU 進行資料並行評估，並使用本地模型
NUM_GPUS=8
MODEL="/mnt/openr1_data_disk/OpenR1_PT/open-r1/data/Qwen2.5-1.5B-Open-R1-Distill-0317"
# 由於 Tesla V100 不支援 bfloat16，這裡使用 half (float16)
MODEL_ARGS="pretrained=$MODEL,dtype=half,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={\"max_new_tokens\":32768,\"temperature\":0.6,\"top_p\":0.95}"
TASK="aime24"
OUTPUT_DIR="data/evals/$(basename $MODEL)"

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

#Qwen2.5-7B-Open-R1-Distill-0317

#!/bin/bash
# 使用 8 個 GPU 進行資料並行評估，並使用本地模型
NUM_GPUS=8
MODEL="/mnt/openr1_data_disk/OpenR1_PT/open-r1/data/Qwen2.5-7B-Open-R1-Distill-0317"
# 由於 Tesla V100 不支援 bfloat16，這裡使用 half (float16)
MODEL_ARGS="pretrained=$MODEL,dtype=half,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={\"max_new_tokens\":32768,\"temperature\":0.6,\"top_p\":0.95}"
TASK="aime24"
OUTPUT_DIR="data/evals/$(basename $MODEL)"

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR