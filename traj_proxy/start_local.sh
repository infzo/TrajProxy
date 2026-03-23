#!/bin/bash
# 本地开发启动脚本
# 用于本地开发环境，连接外部运行的数据库

cd "$(dirname "$0")"

# 设置本地环境变量
export RAY_WORKING_DIR="."
export RAY_PYTHONPATH="."

# 可选：通过环境变量覆盖数据库连接
# export DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"

# 启动 TrajProxy
python -m traj_proxy.app
