#!/bin/bash
source .venv/bin/activate

# ========================
# 多 GPU 訓練參數優化
# ========================
# 硬體參數
NUM_GPUS=8             # 根據實際 GPU 數量調整
GPU_MEM="32GB"         # V100 顯存容量
PRECISION="fp16"       # V100 支援 fp16，bf16 不支援

# 計算參數
BASE_BATCH=8           # 單卡基準批次大小（V100 顯存較小，需降低批次大小）
ACCUM_STEPS=2          # 梯度累積次數（補償批次大小）
MAX_SEQ_LEN=8192       # 序列長度（V100 效能較低，建議降低）

# 動態計算總批次大小
TOTAL_BATCH=$((BASE_BATCH * NUM_GPUS * ACCUM_STEPS))

# 學習率調整（線性擴展規則）
BASE_LR=1.0e-5
ADJUSTED_LR=$(echo "$BASE_LR * $NUM_GPUS * $ACCUM_STEPS" | bc -l)

# ========================
# 進階分散式設定
# ========================
export NCCL_ALGO=Tree          # 使用 Tree 演算法優化 All-Reduce
export NCCL_NSOCKS_PERTHREAD=4 # 提升網路吞吐量
export NCCL_SOCKET_NTHREADS=2  # 增加 Socket 執行緒數

# V100 專用優化
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"  # 非同步記憶體分配
export CUDA_DEVICE_MAX_CONNECTIONS=1                      # V100 PCIe Gen3 頻寬較低

# ========================
# 啟動分散式訓練
# ========================
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  --num_processes $NUM_GPUS \
  --main_process_port 29500 \
  --machine_rank 0 \
  --mixed_precision $PRECISION \
  src/open_r1/sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name open-r1/OpenR1-Math-220k \
  --learning_rate $ADJUSTED_LR \
  --num_train_epochs 5 \
  --packing \
  --max_seq_length $MAX_SEQ_LEN \
  --per_device_train_batch_size $BASE_BATCH \
  --gradient_accumulation_steps $ACCUM_STEPS \
  --gradient_checkpointing \
  --fp16 \
  --output_dir data/Qwen2.5-1.5B-Open-R1-Distill \
  --log_level info \
  --dataloader_num_workers 4 \  # V100 效能較低，減少 dataloader 執行緒數
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{
    "backward_prefetch": "backward_pre",
    "forward_prefetch": true,
    "limit_all_gathers": true,
    "use_orig_params": true
  }'

# ========================
# 監控與除錯建議
# ========================
echo "訓練啟動後建議執行以下監控指令："
echo "1. GPU 使用率監控: watch -n 1 nvidia-smi"
echo "2. 頻寬利用率檢查: nccl-tests/build/all_reduce_perf -b 8G -e 8G -f 2 -g $NUM_GPUS"
echo "3. 訓練進度追蹤: tail -f logs/training.log"