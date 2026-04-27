#!/bin/bash
# 场景 F212: Tokenizer 数据库存储（Proxy 层）
# 测试流程：
#   1. 创建测试 tokenizer（从 models 目录复制）
#   2. 上传 tokenizer 到数据库
#   3. 注册 TITO 模型（使用数据库中的 tokenizer）
#   4. 发送请求验证 tokenizer 自动加载
#   5. 清理：删除模型 + 删除 tokenizer
#
# 数据库连接从 config.yaml 读取，可通过 DATABASE_URL 环境变量覆盖

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils.sh"

echo "========================================"
echo "场景 F212: Tokenizer 数据库存储（Proxy 层）"
echo "========================================"
echo ""

# 测试配置
TEST_TOKENIZER_NAME="test/mock-tokenizer"
TEST_TOKENIZER_DIR="/tmp/test_tokenizer_$$"
TEST_MODEL_NAME="test-tito-model"
TEST_MODEL_RUN_ID="test-run-tito"

# 清理旧数据（防止上次测试失败遗留）
log_info "清理可能存在的旧数据..."
delete_tokenizer "$TEST_TOKENIZER_NAME" > /dev/null 2>&1 || true
curl -s -X DELETE "${API_MODELS}?model_name=${TEST_MODEL_NAME}&run_id=${TEST_MODEL_RUN_ID}" > /dev/null 2>&1 || true

# 清理函数
cleanup() {
    log_info "清理测试资源..."

    # 删除模型
    if [ -n "$MODEL_REGISTERED" ]; then
        curl -s -X DELETE "${API_MODELS}?model_name=${TEST_MODEL_NAME}&run_id=${TEST_MODEL_RUN_ID}" > /dev/null 2>&1 || true
    fi

    # 删除 tokenizer
    if [ -n "$TOKENIZER_UPLOADED" ]; then
        delete_tokenizer "$TEST_TOKENIZER_NAME" > /dev/null 2>&1 || true
    fi

    # 删除临时目录
    rm -rf "$TEST_TOKENIZER_DIR" 2>/dev/null || true
}

trap cleanup EXIT

# ========================================
# 步骤 1: 创建测试 tokenizer
# ========================================
log_step "步骤 1: 创建测试 tokenizer"

TEST_TOKENIZER_PATH=$(create_minimal_test_tokenizer "$TEST_TOKENIZER_DIR")

if [ ! -d "$TEST_TOKENIZER_PATH" ]; then
    log_error "创建测试 tokenizer 失败"
    exit 1
fi

log_response "tokenizer 目录: ${TEST_TOKENIZER_PATH}"
log_response "文件列表: $(ls ${TEST_TOKENIZER_PATH})"
log_separator

# ========================================
# 步骤 2: 上传 tokenizer 到数据库
# ========================================
log_step "步骤 2: 上传 tokenizer 到数据库"

UPLOAD_OUTPUT=$(upload_tokenizer "$TEST_TOKENIZER_NAME" "$TEST_TOKENIZER_PATH")

if [ $? -ne 0 ]; then
    log_error "上传 tokenizer 失败"
    echo "$UPLOAD_OUTPUT"
    exit 1
fi

TOKENIZER_UPLOADED=1
log_response "$UPLOAD_OUTPUT"
log_separator

# 验证上传成功
if ! tokenizer_exists_in_db "$TEST_TOKENIZER_NAME"; then
    log_error "tokenizer 未在数据库中找到"
    exit 1
fi

log_success "tokenizer 已存在于数据库"

# ========================================
# 步骤 3: 注册 TITO 模型
# ========================================
log_step "步骤 3: 注册 TITO 模型（tokenizer_path: ${TEST_TOKENIZER_NAME}）"

log_curl_cmd "curl -s -w '\n%{http_code}' \\
    -X POST '${API_MODELS}/register' \\
    -H 'Content-Type: application/json' \\
    -d '{
        \"run_id\": \"${TEST_MODEL_RUN_ID}\",
        \"model_name\": \"${TEST_MODEL_NAME}\",
        \"url\": \"${BACKEND_MODEL_URL}\",
        \"api_key\": \"${CHAT_API_KEY}\",
        \"tokenizer_path\": \"${TEST_TOKENIZER_NAME}\",
        \"token_in_token_out\": true
    }'"
log_separator

REGISTER_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_MODELS}/register" \
    -H "Content-Type: application/json" \
    -d "{
        \"run_id\": \"${TEST_MODEL_RUN_ID}\",
        \"model_name\": \"${TEST_MODEL_NAME}\",
        \"url\": \"${BACKEND_MODEL_URL}\",
        \"api_key\": \"${CHAT_API_KEY}\",
        \"tokenizer_path\": \"${TEST_TOKENIZER_NAME}\",
        \"token_in_token_out\": true
    }")

REGISTER_BODY=$(echo "$REGISTER_RESPONSE" | sed '$d')
REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | sed -n '$p')

log_response "HTTP Status: ${REGISTER_STATUS}"
log_response "${REGISTER_BODY}"
log_separator

assert_http_status "200" "$REGISTER_STATUS" "HTTP 状态码应为 200"

REGISTER_RESULT=$(json_get "$REGISTER_BODY" "status")
assert_eq "success" "$REGISTER_RESULT" "注册模型应返回 success"

MODEL_REGISTERED=1

echo ""

# ========================================
# 步骤 4: 发送 chat 请求验证 tokenizer 加载
# ========================================
log_step "步骤 4: 发送 chat 请求验证 tokenizer 加载"

log_curl_cmd "curl -s -w '\n%{http_code}' \\
    -X POST '${BASE_URL}/v1/chat/completions' \\
    -H 'Content-Type: application/json' \\
    -H 'Authorization: Bearer ${CHAT_API_KEY}' \\
    -H 'x-run-id: ${TEST_MODEL_RUN_ID}' \\
    -d '{
        \"model\": \"${TEST_MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"hello world\"}]
    }'"
log_separator

CHAT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${CHAT_API_KEY}" \
    -H "x-run-id: ${TEST_MODEL_RUN_ID}" \
    -d "{
        \"model\": \"${TEST_MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"hello world\"}]
    }")

CHAT_BODY=$(echo "$CHAT_RESPONSE" | sed '$d')
CHAT_STATUS=$(echo "$CHAT_RESPONSE" | sed -n '$p')

log_response "HTTP Status: ${CHAT_STATUS}"
log_response "${CHAT_BODY}"
log_separator

assert_http_status "200" "$CHAT_STATUS" "HTTP 状态码应为 200"

echo ""

# ========================================
# 步骤 5: 清理 - 删除模型
# ========================================
log_step "步骤 5: 清理 - 删除模型"

log_curl_cmd "curl -s -w '\n%{http_code}' \\
    -X DELETE '${API_MODELS}?model_name=${TEST_MODEL_NAME}&run_id=${TEST_MODEL_RUN_ID}'"
log_separator

DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "${API_MODELS}?model_name=${TEST_MODEL_NAME}&run_id=${TEST_MODEL_RUN_ID}")

DELETE_BODY=$(echo "$DELETE_RESPONSE" | sed '$d')
DELETE_STATUS=$(echo "$DELETE_RESPONSE" | sed -n '$p')

log_response "HTTP Status: ${DELETE_STATUS}"
log_response "${DELETE_BODY}"
log_separator

assert_http_status "200" "$DELETE_STATUS" "HTTP 状态码应为 200"

MODEL_REGISTERED=""

echo ""

# ========================================
# 步骤 6: 清理 - 删除 tokenizer
# ========================================
log_step "步骤 6: 清理 - 删除 tokenizer"

if delete_tokenizer "$TEST_TOKENIZER_NAME"; then
    log_success "tokenizer 删除成功"
else
    log_warning "tokenizer 删除可能失败"
fi

TOKENIZER_UPLOADED=""

echo ""

# ========================================
# 步骤 7: 验证清理完成
# ========================================
log_step "步骤 7: 验证清理完成"

if tokenizer_exists_in_db "$TEST_TOKENIZER_NAME"; then
    log_error "tokenizer 仍在数据库中，清理不完整"
    exit 1
else
    log_success "tokenizer 已从数据库删除"
fi

echo ""

# 打印测试摘要
print_summary
