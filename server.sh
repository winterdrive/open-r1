#!/bin/bash
# ==============================================
# OpenR1 專案環境自動化設定腳本
# 版本：2.0 (結構優化版)
# ==============================================

# --------------------------
# 前置系統檢查與環境清理
# --------------------------
echo "[1/7] 正在執行系統環境檢查..."
# 檢查是否為 Ubuntu 20.04/22.04
if ! grep -qE 'Ubuntu (20.04|22.04)' /etc/issue; then
    echo "[錯誤] 僅支援 Ubuntu 20.04/22.04 系統"
    exit 1
fi
# 檢查是否為 NVIDIA GPU 環境
echo lspci | grep -i nvidia || {
    echo "[錯誤] 無法偵測到 NVIDIA GPU 裝置"
    exit 1
}

# 清理可能存在的衝突環境
echo "[2/7] 正在清除舊版CUDA環境..."
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" \
    "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "*cuda*" "*nvidia*" -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y

# --------------------------
# CUDA 12.4 完整安裝流程
# --------------------------
echo "[3/7] 正在安裝CUDA 12.4..."
# 下載官方安裝套件
wget -q https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# 執行靜默安裝（含驅動程式）
sudo sh cuda_12.4.0_550.54.14_linux.run --override --silent --toolkit --samples --driver

# 永久化環境變數
cat <<EOF | sudo tee /etc/profile.d/cuda.sh > /dev/null
export PATH=/usr/local/cuda-12.4/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH
EOF
source /etc/profile

# --------------------------
# 專案基礎環境建置
# --------------------------
echo "[4/7] 正在初始化專案環境..."
# 建立隔離工作區
mkdir -p OpenR1_PT && cd OpenR1_PT

# 複製程式碼儲存庫（使用淺層複製節省頻寬）
git clone --depth=1 https://github.com/huggingface/open-r1.git
cd open-r1

# 建立Python虛擬環境
python3 -m venv .venv --prompt "OpenR1-Env"
source .venv/bin/activate

# --------------------------
# PyTorch 生態系安裝
# --------------------------
echo "[5/7] 正在安裝PyTorch生態系..."
# 基礎工具鏈
python -m pip install --upgrade pip wheel setuptools

# 安裝PyTorch（預編譯CUDA版本）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 安裝加速元件
pip install flash-attn==2.5.8 --no-build-isolation  # 使用預編譯二進位檔
pip install vllm==0.7.2  # 指定相容版本

# --------------------------
# 專案相依套件與工具鏈
# --------------------------
echo "[6/7] 正在設定開發工具鏈..."
# 安裝編譯相依套件
sudo apt-get install -y build-essential ninja-build cmake

# 安裝版本管控工具
sudo apt-get install -y git-lfs
git lfs install

# 安裝專案相依套件（開發模式）
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"

# --------------------------
# 模型訓練啟動設定
# --------------------------
echo "[7/7] 正在準備訓練環境..."
# 服務認證（需手動操作）
huggingface-cli login
wandb login

# 分散式訓練啟動範本
cat <<EOF > train.sh
#!/bin/bash
source .venv/bin/activate

accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml \\
    src/open_r1/sft.py \\
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \\
    --dataset_name open-r1/OpenR1-Math-220k \\
    --learning_rate 1.0e-5 \\
    --num_train_epochs 1 \\
    --packing \\
    --max_seq_length 16384 \\
    --per_device_train_batch_size 16 \\
    --gradient_checkpointing \\
    --bf16 \\
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
EOF

chmod +x train.sh

# ==============================================
# 環境驗證指令
# ==============================================
echo "安裝完成！請依序執行以下驗證指令："
echo "1. GPU 基礎驗證: nvidia-smi -L"
echo "2. CUDA 工具鏈驗證: nvcc --version"
echo "3. PyTorch CUDA 支援驗證: python -c \"import torch; print(torch.cuda.is_available())\""
echo "4. 啟動訓練任務: ./train.sh"