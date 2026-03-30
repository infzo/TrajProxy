#!/bin/bash
# 本地开发启动脚本
# 用于本地开发环境，连接外部运行的数据库

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 设置本地环境变量
export RAY_WORKING_DIR="."
export RAY_PYTHONPATH="."

# 启动 TrajProxy
python -m traj_proxy.app
