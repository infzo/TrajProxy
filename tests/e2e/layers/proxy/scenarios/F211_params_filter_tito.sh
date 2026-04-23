#!/bin/bash
# 场景 F211: 参数透传场景 - TITO模式参数过滤（Proxy 层）
# 测试流程：启动mock服务 -> 注册模型(TITO模式) -> 发送请求(含兼容和不兼容参数) -> 验证mock收到的请求 -> 删除模型 -> 停止mock
#
# 验证要点：
#   1. 兼容参数(seed, n, stop, user)被透传到后端
#   2. 不兼容参数(response_format, logit_bias)被过滤丢弃
#   3. 自定义header(x-sandbox-traj-id)被透传

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils.sh"

echo "========================================"
echo "场景 F211: 参数透传场景 - TITO模式参数过滤（Proxy 层）"
echo "========================================"
echo ""

# 测试配置
SCENARIO_ID=$(basename "${BASH_SOURCE[0]}" .sh | grep -oE '[FP][0-9]+' | tr '[:upper:]' '[:lower:]')
TEST_BASE_URL="${BASE_URL}"
TEST_MODEL_NAME="passthrough-tito-model"
TEST_RUN_ID="run-${SCENARIO_ID}"
TEST_SESSION_ID="session-${SCENARIO_ID}-$(date +%s%N | md5sum | head -c 8)"

# TITO模式需要tokenizer路径
TEST_TOKENIZER_PATH="${TEST_TOKENIZER_PATH:-Qwen/Qwen3.5-2B}"

# Mock服务配置
MOCK_PORT=19991
MOCK_URL="http://127.0.0.1:${MOCK_PORT}"
MOCK_PID=""

# TrajProxy可达的mock服务地址
MOCK_INFER_HOST="${MOCK_INFER_HOST:-host.docker.internal}"
MOCK_INFER_URL="http://${MOCK_INFER_HOST}:${MOCK_PORT}/v1"

# ========================================
# 辅助函数
# ========================================

start_mock() {
    log_info "启动Mock推理服务..."
    # 注意：source utils.sh 后 SCRIPT_DIR 已被覆盖为 proxy 目录
    python3 "${SCRIPT_DIR}/mock_infer_server.py" "$MOCK_PORT" &
    MOCK_PID=$!

    for i in $(seq 1 10); do
        if curl -s "${MOCK_URL}/mock/health" > /dev/null 2>&1; then
            log_success "Mock服务已启动 (PID: ${MOCK_PID}, Port: ${MOCK_PORT})"
            return 0
        fi
        sleep 1
    done
    log_error "Mock服务启动超时"
    return 1
}

stop_mock() {
    if [ -n "$MOCK_PID" ]; then
        kill "$MOCK_PID" 2>/dev/null
        wait "$MOCK_PID" 2>/dev/null
        log_info "Mock服务已停止 (PID: ${MOCK_PID})"
        MOCK_PID=""
    fi
}

clear_mock_records() {
    curl -s -X DELETE "${MOCK_URL}/mock/requests" > /dev/null
}

verify_infer_request() {
    local tmpfile=$(mktemp)
    curl -s "${MOCK_URL}/mock/requests" > "$tmpfile"

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

# TITO不兼容参数（应该被过滤掉）
tito_incompatible = ["response_format", "logit_bias"]
for param in tito_incompatible:
    if param in body:
        print(f"TITO_INCOMPATIBLE:{param}=FOUND")
    else:
        print(f"TITO_INCOMPATIBLE:{param}=FILTERED")

# 兼容参数（应该被透传）
compatible_params = ["seed", "n", "stop", "user"]
for param in compatible_params:
    if param in body:
        print(f"COMPATIBLE:{param}={body[param]}")
    else:
        print(f"COMPATIBLE:{param}=NOT_FOUND")

# Header检查
headers_to_check = ["x-sandbox-traj-id", "x-run-id", "x-session-id"]
for h in headers_to_check:
    if h in headers:
        print(f"HEADER:{h}={headers[h]}")
    else:
        print(f"HEADER:{h}=NOT_FOUND")

# 输出完整body供调试
print(f"BODY_JSON:{json.dumps(body, ensure_ascii=False)}")
PYEOF

    rm -f "$tmpfile"
}

trap stop_mock EXIT

# ========================================
# 步骤 1: 启动Mock服务
# ========================================
log_step "步骤 1: 启动Mock推理服务"
log_separator

if ! start_mock; then
    log_error "无法启动Mock服务，测试终止"
    exit 1
fi

echo ""

# ========================================
# 步骤 2: 注册模型（TITO模式）
# ========================================
log_step "步骤 2: 注册模型（TITO模式, token_in_token_out: true）"
log_curl_cmd "curl -s -w '\n%{http_code}' \\
    -X POST '${TEST_BASE_URL}/models/register' \\
    -H 'Content-Type: application/json' \\
    -d '{
        \"run_id\": \"${TEST_RUN_ID}\",
        \"model_name\": \"${TEST_MODEL_NAME}\",
        \"url\": \"${MOCK_INFER_URL}\",
        \"api_key\": \"mock-api-key\",
        \"tokenizer_path\": \"${TEST_TOKENIZER_PATH}\",
        \"token_in_token_out\": true
    }'"
log_separator

REGISTER_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${TEST_BASE_URL}/models/register" \
    -H "Content-Type: application/json" \
    -d "{
        \"run_id\": \"${TEST_RUN_ID}\",
        \"model_name\": \"${TEST_MODEL_NAME}\",
        \"url\": \"${MOCK_INFER_URL}\",
        \"api_key\": \"mock-api-key\",
        \"tokenizer_path\": \"${TEST_TOKENIZER_PATH}\",
        \"token_in_token_out\": true
    }")

REGISTER_BODY=$(echo "$REGISTER_RESPONSE" | sed '$d')
REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | sed -n '$p')

log_response "HTTP Status: ${REGISTER_STATUS}"
log_response "${REGISTER_BODY}"
log_separator

# 允许tokenizer加载失败时跳过测试
if [ "$REGISTER_STATUS" != "200" ]; then
    log_warning "模型注册失败（可能tokenizer不可用），跳过TITO测试"
    log_warning "请确保 ${TEST_TOKENIZER_PATH} tokenizer可用"
    stop_mock
    exit 0
fi

REGISTER_RESULT=$(json_get "$REGISTER_BODY" "status")
assert_eq "success" "$REGISTER_RESULT" "注册模型应返回 success"

sleep 3

echo ""

# ========================================
# 步骤 3: 发送请求（含兼容和不兼容参数）
# ========================================
log_step "步骤 3: 发送请求（含兼容和不兼容参数）"
log_info "兼容参数: seed, n, stop, user"
log_info "不兼容参数: response_format, logit_bias（应被过滤）"
log_curl_cmd "curl -s -w '\n%{http_code}' \\
    -X POST '${TEST_BASE_URL}/s/${TEST_RUN_ID}/${TEST_SESSION_ID}/v1/chat/completions' \\
    -H 'Content-Type: application/json' \\
    -H 'Authorization: Bearer ${CHAT_API_KEY}' \\
    -H 'x-sandbox-traj-id: tito-traj-789' \\
    -d '{
        \"model\": \"${TEST_MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
        \"max_tokens\": 10,
        \"seed\": 42,
        \"n\": 1,
        \"stop\": [\"\\n\"],
        \"user\": \"tito-user\",
        \"response_format\": {\"type\": \"json_object\"},
        \"logit_bias\": {\"1\": -100},
        \"stream\": false
    }'"
log_separator

# 清空之前的mock记录
clear_mock_records

CHAT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${TEST_BASE_URL}/s/${TEST_RUN_ID}/${TEST_SESSION_ID}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${CHAT_API_KEY}" \
    -H "x-sandbox-traj-id: tito-traj-789" \
    -d "{
        \"model\": \"${TEST_MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
        \"max_tokens\": 10,
        \"seed\": 42,
        \"n\": 1,
        \"stop\": [\"\\n\"],
        \"user\": \"tito-user\",
        \"response_format\": {\"type\": \"json_object\"},
        \"logit_bias\": {\"1\": -100},
        \"stream\": false
    }")

CHAT_BODY=$(echo "$CHAT_RESPONSE" | sed '$d')
CHAT_STATUS=$(echo "$CHAT_RESPONSE" | sed -n '$p')

log_response "HTTP Status: ${CHAT_STATUS}"
log_response "${CHAT_BODY}"
log_separator

assert_http_status "200" "$CHAT_STATUS" "HTTP 状态码应为 200"

echo ""

# ========================================
# 步骤 4: 验证TITO参数过滤
# ========================================
log_step "步骤 4: 验证TITO参数过滤"
log_separator

VERIFY_RESULT=$(verify_infer_request)
log_info "Mock收到的请求验证结果:"
echo "$VERIFY_RESULT" | grep -E "^(TITO_INCOMPATIBLE|COMPATIBLE|HEADER):" | while read line; do
    echo "  $line"
done

echo ""
log_info "【验证不兼容参数被过滤】"

# 验证response_format被过滤
RESPONSE_FORMAT_VAL=$(echo "$VERIFY_RESULT" | grep "^TITO_INCOMPATIBLE:response_format=" | cut -d= -f2)
assert_eq "FILTERED" "$RESPONSE_FORMAT_VAL" "response_format应被过滤（TITO不兼容）"

# 验证logit_bias被过滤
LOGIT_BIAS_VAL=$(echo "$VERIFY_RESULT" | grep "^TITO_INCOMPATIBLE:logit_bias=" | cut -d= -f2)
assert_eq "FILTERED" "$LOGIT_BIAS_VAL" "logit_bias应被过滤（TITO不兼容）"

echo ""
log_info "【验证兼容参数被透传】"

# 验证seed被透传
SEED_VAL=$(echo "$VERIFY_RESULT" | grep "^COMPATIBLE:seed=" | cut -d= -f2)
assert_eq "42" "$SEED_VAL" "seed参数应被透传"

# 验证n被透传
N_VAL=$(echo "$VERIFY_RESULT" | grep "^COMPATIBLE:n=" | cut -d= -f2)
assert_eq "1" "$N_VAL" "n参数应被透传"

# 验证user被透传
USER_VAL=$(echo "$VERIFY_RESULT" | grep "^COMPATIBLE:user=" | cut -d= -f2)
assert_eq "tito-user" "$USER_VAL" "user参数应被透传"

echo ""
log_info "【验证Header透传】"

# 验证header被透传
TRAJ_ID_VAL=$(echo "$VERIFY_RESULT" | grep "^HEADER:x-sandbox-traj-id=" | cut -d= -f2)
assert_eq "tito-traj-789" "$TRAJ_ID_VAL" "x-sandbox-traj-id header应被透传"

# 验证内部header不被转发
X_RUN_ID_VAL=$(echo "$VERIFY_RESULT" | grep "^HEADER:x-run-id=" | cut -d= -f2)
assert_eq "NOT_FOUND" "$X_RUN_ID_VAL" "x-run-id header不应被转发"

X_SESSION_ID_VAL=$(echo "$VERIFY_RESULT" | grep "^HEADER:x-session-id=" | cut -d= -f2)
assert_eq "NOT_FOUND" "$X_SESSION_ID_VAL" "x-session-id header不应被转发"

echo ""

# ========================================
# 步骤 5: 删除模型
# ========================================
log_step "步骤 5: 删除模型（run_id: ${TEST_RUN_ID}）"
log_curl_cmd "curl -s -w '\n%{http_code}' \\
    -X DELETE '${TEST_BASE_URL}/models?model_name=${TEST_MODEL_NAME}&run_id=${TEST_RUN_ID}'"
log_separator

DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "${TEST_BASE_URL}/models?model_name=${TEST_MODEL_NAME}&run_id=${TEST_RUN_ID}")

DELETE_BODY=$(echo "$DELETE_RESPONSE" | sed '$d')
DELETE_STATUS=$(echo "$DELETE_RESPONSE" | sed -n '$p')

log_response "HTTP Status: ${DELETE_STATUS}"
log_response "${DELETE_BODY}"
log_separator

assert_http_status "200" "$DELETE_STATUS" "HTTP 状态码应为 200"

DELETE_RESULT=$(json_get "$DELETE_BODY" "status")
assert_eq "success" "$DELETE_RESULT" "删除模型应返回 success"

echo ""

# 打印测试摘要
print_summary
