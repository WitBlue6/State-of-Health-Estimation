#!/bin/bash

# conda 初始化
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 文件名和下载地址
WHL_FILE="torch_npu-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl"
DOWNLOAD_URL="https://gitee.com/ascend/pytorch/releases/download/v7.1.0-pytorch2.6.0/torch_npu-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl"

echo "安装conda环境"

conda create -n soh python=3.10
conda init

echo "激活conda环境<soh>"
conda activate soh

# 1 基础深度学习库
echo "当前 Python 路径: $(which python)"
echo "切换 pip 源到清华镜像"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装官方pytorch"
python -m pip install numpy==1.26.0 
python -m pip install torch==2.6.0 

# 检查文件是否存在
if [ -f "$WHL_FILE" ]; then
    echo "$WHL_FILE 已存在，跳过下载。"
else
    echo "$WHL_FILE 不存在，开始下载..."
    wget "$DOWNLOAD_URL" -O "$WHL_FILE"
    if [ $? -ne 0 ]; then
        echo "下载失败，请检查网络或 URL。"
        exit 1
    fi
fi

# 安装 whl 文件
echo "安装 $WHL_FILE ..."
python -m pip install "$WHL_FILE"

python -m pip install torchvision==0.21.0

# 2 LLM依赖库
echo "安装LLM依赖库"
python -m pip install transformers==4.56.1 --no-deps
python -m pip install sentence_transformers==5.1.0 --no-deps

# 3 其他依赖库
echo "安装其他依赖库"
python -m pip install openai chromadb scikit-learn pandas matplotlib websockets

# 4 剩余依赖库
python -m pip install regex safetensors decorator psutil
