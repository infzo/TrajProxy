#!/bin/bash
# Layer 2 utils 包装: 加载共享 utils + proxy 层配置

_LAYER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_LAYER_DIR}/../../utils.sh"
source "${_LAYER_DIR}/config.sh"

# Mock推理服务脚本路径
MOCK_INFER_SERVER="${_LAYER_DIR}/mock_infer_server.py"

# TrajProxy可达的mock服务地址（Docker环境用host.docker.internal）
MOCK_INFER_HOST="${MOCK_INFER_HOST:-host.docker.internal}"

# 项目根目录 (tests/e2e/layers/proxy -> 项目根需要跳 4 级)
PROJECT_ROOT="$(cd "${_LAYER_DIR}/../../../.." && pwd)"

# ========================================
# Mock推理服务辅助函数
# ========================================

# 启动Mock推理服务
# 使用前需设置: MOCK_PORT, MOCK_URL
start_mock() {
    log_info "启动Mock推理服务..."

    # 清理占用端口的残留进程
    local occupying_pid
    occupying_pid=$(lsof -ti :"$MOCK_PORT" 2>/dev/null)
    if [ -n "$occupying_pid" ]; then
        log_warning "端口 $MOCK_PORT 被残留进程 (PID: $occupying_pid) 占用，正在清理..."
        kill -9 $occupying_pid 2>/dev/null
        sleep 1
    fi

    python3 "${MOCK_INFER_SERVER}" "$MOCK_PORT" &
    MOCK_PID=$!

    # 等待mock服务启动
    local health_result
    for i in $(seq 1 10); do
        # 检查进程是否已退出（端口冲突等情况）
        if ! kill -0 "$MOCK_PID" 2>/dev/null; then
            log_error "Mock服务进程异常退出（可能端口 $MOCK_PORT 被占用）"
            MOCK_PID=""
            return 1
        fi

        health_result=$(curl -s --noproxy '*' --max-time 3 "${MOCK_URL}/mock/health" 2>&1)
        # 只检查响应内容是否包含 "ok"，不依赖 curl 退出码
        # （某些环境下 curl 可能返回数据但退出码非零）
        if echo "$health_result" | grep -q "ok"; then
            log_success "Mock服务已启动 (PID: ${MOCK_PID}, Port: ${MOCK_PORT})"
            return 0
        fi
        sleep 1
    done

    # 超时时输出诊断信息
    log_error "Mock服务启动超时"
    log_error "进程状态: $(kill -0 $MOCK_PID 2>&1 && echo "运行中" || echo "已退出")"
    log_error "端口状态: $(lsof -i :$MOCK_PORT 2>/dev/null | head -1 || echo "未被监听")"
    log_error "健康检查结果: $health_result"

    # 清理启动失败的进程
    kill "$MOCK_PID" 2>/dev/null
    wait "$MOCK_PID" 2>/dev/null
    MOCK_PID=""
    return 1
}

# 停止Mock推理服务
stop_mock() {
    if [ -n "$MOCK_PID" ]; then
        kill "$MOCK_PID" 2>/dev/null
        wait "$MOCK_PID" 2>/dev/null
        log_info "Mock服务已停止 (PID: ${MOCK_PID})"
        MOCK_PID=""
    fi
}

# 清空Mock服务的请求记录
clear_mock_records() {
    curl -s --noproxy '*' --max-time 5 -X DELETE "${MOCK_URL}/mock/requests" > /dev/null
}

# 从Mock服务获取推理请求记录
# 参数: $1 - 要检查的body参数列表(空格分隔)
#        $2 - 要检查的header列表(空格分隔)
# 输出: BODY:param=value / HEADER:header=value 格式
verify_infer_request() {
    local body_params="${1:-}"
    local header_params="${2:-}"
    local tmpfile
    tmpfile=$(mktemp)
    curl -s --noproxy '*' --max-time 5 "${MOCK_URL}/mock/requests" > "$tmpfile"

    python3 << PYEOF
import json
import sys

with open("$tmpfile", "r") as f:
    data = json.load(f)

# 找到最后一个推理请求
infer_req = None
for req in reversed(data.get("requests", [])):
    if req["path"] in ["/v1/chat/completions", "/v1/completions"]:
        infer_req = req
        break

if infer_req is None:
    print("ERROR:no_infer_request")
    sys.exit(1)

body = infer_req.get("body", {})
headers = {k.lower(): v for k, v in infer_req.get("headers", {}).items()}

# 输出body参数
for param in "$body_params".split():
    if param in body:
        print(f"BODY:{param}={body[param]}")
    else:
        print(f"BODY:{param}=NOT_FOUND")

# 输出header信息
for h in "$header_params".split():
    if h in headers:
        print(f"HEADER:{h}={headers[h]}")
    else:
        print(f"HEADER:{h}=NOT_FOUND")

# 输出完整body供调试
print(f"BODY_JSON:{json.dumps(body, ensure_ascii=False)}")
PYEOF

    rm -f "$tmpfile"
}

# ========================================
# Tokenizer 数据库存储辅助函数
# ========================================

# 获取测试用的数据库 URL
# 优先级: TEST_DATABASE_URL 环境变量 > Docker 默认连接
get_test_db_url() {
    if [ -n "${TEST_DATABASE_URL:-}" ]; then
        echo "$TEST_DATABASE_URL"
    else
        # Docker 环境默认数据库连接
        echo "postgresql://llmproxy:dbpassword9090@127.0.0.1:5432/traj_proxy"
    fi
}

# 上传 tokenizer 到数据库
# 参数: $1 - name (如 "test/mock-tokenizer")
#       $2 - local_path (本地 tokenizer 目录)
# 返回: 0 成功, 1 失败
# 说明: 数据库连接从 TEST_DATABASE_URL 或默认 Docker 连接读取
upload_tokenizer() {
    local name="$1"
    local local_path="$2"
    local db_url
    db_url=$(get_test_db_url)

    log_info "上传 tokenizer: ${name} <- ${local_path}"

    local result
    # 使用 perl alarm 实现超时保护（macOS 没有 timeout 命令）
    result=$(perl -e 'alarm 30; exec @ARGV' -- \
        python3 "${PROJECT_ROOT}/scripts/manage_tokenizer.py" upload \
        --name "$name" \
        --path "$local_path" \
        --db-url "$db_url" 2>&1)

    local exit_code=$?

    if [ $exit_code -eq 0 ] && echo "$result" | grep -q "上传成功"; then
        log_success "tokenizer 上传成功: ${name}"
        echo "$result"
        return 0
    elif [ $exit_code -eq 142 ]; then
        # SIGALRM (142 = 128 + 14)，表示超时
        log_error "tokenizer 上传超时: ${name}（数据库连接超时）"
        echo "$result"
        return 1
    else
        log_error "tokenizer 上传失败: ${name}"
        echo "$result"
        return 1
    fi
}

# 列出数据库中的 tokenizer
# 输出: JSON 格式的列表结果
list_tokenizers() {
    local db_url
    db_url=$(get_test_db_url)
    # 使用 perl alarm 实现超时保护（macOS 没有 timeout 命令）
    perl -e 'alarm 30; exec @ARGV' -- \
        python3 "${PROJECT_ROOT}/scripts/manage_tokenizer.py" list \
        --db-url "$db_url" 2>&1
}

# 删除数据库中的 tokenizer
# 参数: $1 - name
# 返回: 0 成功, 1 失败
delete_tokenizer() {
    local name="$1"
    local db_url
    db_url=$(get_test_db_url)

    log_info "删除 tokenizer: ${name}"

    local result
    # 使用 perl alarm 实现超时保护（macOS 没有 timeout 命令）
    result=$(perl -e 'alarm 30; exec @ARGV' -- \
        python3 "${PROJECT_ROOT}/scripts/manage_tokenizer.py" delete \
        --name "$name" \
        --db-url "$db_url" 2>&1)

    local exit_code=$?

    if [ $exit_code -eq 0 ] && echo "$result" | grep -q "删除成功"; then
        log_success "tokenizer 删除成功: ${name}"
        return 0
    elif [ $exit_code -eq 142 ]; then
        # SIGALRM (142 = 128 + 14)，表示超时
        log_error "tokenizer 删除超时: ${name}（数据库连接超时）"
        echo "$result"
        return 1
    else
        log_error "tokenizer 删除失败: ${name}"
        echo "$result"
        return 1
    fi
}

# 检查 tokenizer 是否存在于数据库
# 参数: $1 - name
# 返回: 0 存在, 1 不存在
tokenizer_exists_in_db() {
    local name="$1"
    local list_output
    list_output=$(list_tokenizers)

    if echo "$list_output" | grep -q "$name"; then
        return 0
    else
        return 1
    fi
}

# 创建最小测试 tokenizer 目录
# 参数: $1 - 输出目录路径
# 返回: tokenizer 目录路径 (stdout)，日志输出到 stderr
# 说明: 从 models/Qwen/Qwen3.5-2B-TITO 复制真实 tokenizer 文件
create_minimal_test_tokenizer() {
    local output_dir="$1"
    local tokenizer_dir="${output_dir}/test-tokenizer"
    local source_tokenizer="${PROJECT_ROOT}/models/Qwen/Qwen3.5-2B-TITO"

    log_info "创建测试 tokenizer: ${tokenizer_dir}" >&2

    # 检查源 tokenizer 是否存在
    if [ ! -d "$source_tokenizer" ]; then
        log_error "源 tokenizer 不存在: ${source_tokenizer}" >&2
        return 1
    fi

    # 创建目标目录
    mkdir -p "$tokenizer_dir"

    # 复制必要的 tokenizer 文件
    cp "${source_tokenizer}/tokenizer.json" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/tokenizer_config.json" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/special_tokens_map.json" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/merges.txt" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/vocab.json" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/config.json" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/preprocessor_config.json" "$tokenizer_dir/" 2>/dev/null || true
    cp "${source_tokenizer}/generation_config.json" "$tokenizer_dir/" 2>/dev/null || true

    # 验证必要文件存在
    if [ ! -f "${tokenizer_dir}/tokenizer.json" ]; then
        log_error "tokenizer.json 复制失败" >&2
        return 1
    fi

    log_success "测试 tokenizer 创建完成: ${tokenizer_dir}" >&2
    echo "$tokenizer_dir"
}
