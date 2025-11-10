#!/bin/bash

# 前端项目路径
FRONTEND_DIR="/home/lzh/dashboard"
# 后端项目路径
BACKEND_DIR="/home/lzh/soh_classification"

# 检查nginx是否正在运行
check_nginx_running() {
    # 使用pgrep检查nginx进程是否存在
    if pgrep -x "nginx" > /dev/null; then
        return 0  # nginx正在运行
    else
        return 1  # nginx未运行
    fi
}

# 1. 构建项目
echo "开始部署项目..."

# 2. 检查前端目录是否存在
cd $FRONTEND_DIR || { echo "前端目录不存在"; exit 1; }

# 3. 检查nginx是否正在运行，如果没有运行则启动nginx
if check_nginx_running; then
    echo "nginx已经在运行"
else
    echo "nginx未运行，正在启动nginx..."
    sudo nginx
    if [ $? -eq 0 ]; then
        echo "nginx启动成功"
    else
        echo "nginx启动失败，请检查配置"
        exit 1
    fi
fi


# 4. 检查后端目录是否存在
cd $BACKEND_DIR || { echo "后端目录不存在"; exit 1; }

# 5. 启动后端服务
nohup python3.10 ws.py 2>&1 &

echo "部署完成！"
