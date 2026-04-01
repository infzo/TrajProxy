#!/bin/bash
# TrajProxy 容器启动入口脚本
# 启动应用前自动检测并初始化数据库（幂等操作）

set -e

echo "=== TrajProxy 容器启动 ==="

# 等待 PostgreSQL 就绪（最多重试 30 次，每次间隔 2 秒）
if [ -n "${DATABASE_URL}" ]; then
    echo "[init_db] 检测数据库连接..."
    MAX_RETRIES=30
    RETRY_INTERVAL=2
    for i in $(seq 1 $MAX_RETRIES); do
        if python /app/scripts/init_db.py --db-url "${DATABASE_URL}" 2>&1; then
            echo "[init_db] 数据库初始化完成"
            break
        fi
        echo "[init_db] 第 ${i}/${MAX_RETRIES} 次重试，${RETRY_INTERVAL} 秒后重试..."
        sleep ${RETRY_INTERVAL}
    done
else
    echo "[init_db] 未设置 DATABASE_URL 环境变量，跳过数据库初始化"
fi

echo "=== 启动 TrajProxy 服务 ==="
exec python -m traj_proxy.app
