```shell
# 注册模型 run id
curl -X POST http://localhost:12300/models/register \
  -H "Content-Type: application/json" \
  -d '{
  "run_id": "ma-job-proxy-test2",
  "model_name": "/Temp/bucket-siye-green-guiyang-code/MindForge-Coder/models/Qwen3-Coder-30B-A3B-Instruct",
  "url": "http://7.242.105.47:8206",
  "api_key": "sk-1234"
}'

# 查询模型
curl http://localhost:12300/models

# 下发请求
curl -X POST http://localhost:12300/s/ma-job-proxy-test2,sample_001,task_001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "/Temp/bucket-siye-green-guiyang-code/MindForge-Coder/models/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'

# 下发请求
curl -X POST http://localhost:12300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: ma-job-proxy-test2,sample_001,task_001" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "/Temp/bucket-siye-green-guiyang-code/MindForge-Coder/models/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'

# 删除模型
curl -X DELETE http://localhost:12300/models?model_name=/Temp/bucket-siye-green-guiyang-code/MindForge-Coder/models/Qwen3-Coder-30B-A3B-Instruct


# 注册模型
curl -X POST http://localhost:12300/models/register \
  -H "Content-Type: application/json" \
  -d '{
  "model_name": "/Temp/bucket-siye-green-guiyang-code/MindForge-Coder/models/Qwen3-Coder-30B-A3B-Instruct",
  "url": "http://7.242.105.47:8206",
  "api_key": "sk-1234"
}'

# 下发请求
curl -X POST http://localhost:12300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: ,sample_001,task_001" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "/Temp/bucket-siye-green-guiyang-code/MindForge-Coder/models/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'

```


今天(2026-04-02)的Commit修改总结                                                          
                                         
  1. b50890f - feat: 添加测试用例                                                           
   
  - 删除旧的 manual_test 文档和脚本，新增 tests/verify 目录下的完整测试验证框架             
                                         
  2. 999cb46 - fix: 数据库统一public前缀                                                    
                                         
  - 统一数据库表前缀为 public，更新仓库层代码和 Docker 配置                                 
                                         
  3. 4892092 - fix: xx                                                                      
                                         
  - 重构 run-id 和 model-name 逻辑，新增验证器和规则文档                                    
                                         
  4. 3aad185 - feat: 重构run-id和model-name逻辑                                             
                                         
  - 进一步优化 run-id 和 model-name 的处理逻辑                                              
                                         
  5. e49a1cd - fix: 优化注册、列出模型接口                                                  
                                         
  - 优化模型注册和列表接口的实现     
