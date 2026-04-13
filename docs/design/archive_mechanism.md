# 数据库归档机制设计文档

> **导航**: [文档中心](../README.md) | [数据库设计](database.md)

---

## 1. 核心设计理念

### 1.1 问题背景

请求轨迹数据的典型特征：

| 数据类型 | 增长速度 | 访问频率 | 存储占比 |
|----------|----------|----------|----------|
| 统计元数据 | 慢 | 高 | ~5% |
| 详情大字段 | 快 | 低 | ~95% |

**核心矛盾**：详情大字段增长快、访问少，却占据了绝大部分存储空间。

### 1.2 设计目标

```
┌─────────────────────────────────────────────────────────────┐
│                      设计目标金字塔                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                       ┌─────────┐                           │
│                       │ 零运维  │ ← 应用内调度，无外部依赖     │
│                       └────┬────┘                           │
│                    ┌───────┴───────┐                        │
│                    │   零 VACUUM   │ ← 分区删除，空间立回收   │
│                    └───────┬───────┘                        │
│              ┌─────────────┴─────────────┐                  │
│              │  统计能力不受影响（元数据） │                  │
│              └─────────────┬─────────────┘                  │
│        ┌───────────────────┴───────────────────┐            │
│        │  详情数据可归档可恢复（JSONL+GZIP）    │            │
│        └───────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 整体架构

### 2.1 数据分层架构

```mermaid
graph TB
    subgraph 写入层["写入层"]
        ctx[ProcessContext] --> repo[RequestRepository.insert]
        repo --> |双写事务| meta & detail
    end

    subgraph 存储层["存储层"]
        meta[request_metadata<br/>元数据表<br/>━━━━━━━━━━━<br/>长期保留<br/>含统计信息] 
        detail[request_details_active<br/>详情分区表<br/>━━━━━━━━━━━<br/>按月分区<br/>仅存近期]
        
        meta --> |"archive_location = NULL"| detail
        meta -.-> |"archive_location = 文件名"| archive_file
    end

    subgraph 归档层["归档层"]
        scheduler[ArchiveScheduler<br/>定时调度器] --> |触发| archiver[归档执行器]
        archiver --> |导出| archive_file[/data/archives/<br/>YYYY_MM.jsonl.gz]
        archiver --> |更新| meta
        archiver --> |删除| detail
    end

    subgraph 查询层["查询层"]
        query_active[活跃数据查询] --> |JOIN| meta & detail
        query_archived[已归档查询] --> |仅查| meta
        query_archived -.-> |读文件| archive_file
    end

    style meta fill:#e1f5fe
    style detail fill:#fff3e0
    style archive_file fill:#e8f5e9
    style scheduler fill:#fce4ec
```

### 2.2 表职责划分

| 表名 | 职责 | 生命周期 | 关键字段 |
|------|------|----------|----------|
| `request_metadata` | 存储统计信息，关联查询入口 | **永久保留** | `archive_location` |
| `request_details_active` | 存储详情大字段 | **按月分区，过期归档** | `created_at`（分区键） |

**关键字段 `archive_location` 状态转移**：

```mermaid
stateDiagram-v2
    [*] --> 活跃: 数据写入
    活跃 --> 已归档: 归档完成
    已归档 --> [*]: 永久标记
    
    note right of 活跃
        archive_location = NULL
        详情在分区表中
    end note
    
    note right of 已归档
        archive_location = "2026_03.jsonl.gz"
        详情在外部文件
    end note
```

---

## 3. 分区管理机制

### 3.1 分区结构

```mermaid
graph LR
    subgraph request_details_active["request_details_active (父表)"]
        direction LR
        P_2026_02["2026_02<br/>━━━━━━━<br/>[2/1, 3/1)"]
        P_2026_03["2026_03<br/>━━━━━━━<br/>[3/1, 4/1)"]
        P_2026_04["2026_04<br/>━━━━━━━<br/>[4/1, 5/1)<br/>当前月"]
        P_default["default<br/>━━━━━━━<br/>兜底分区"]
    end

    style P_2026_04 fill:#c8e6c9
    style P_2026_02 fill:#ffcdd2
    style P_2026_03 fill:#ffcdd2
```

**分区命名规范**：`request_details_active_YYYY_MM`

### 3.2 分区自动创建

```mermaid
flowchart TD
    start[应用启动 / 归档执行前] --> check{检查当月分区}
    check --> |不存在| create_current[创建当月分区]
    check --> |存在| check_next{检查下月分区}
    create_current --> check_next
    
    check_next --> |不存在| create_next[创建下月分区]
    check_next --> |存在| check_default{检查默认分区}
    create_next --> check_default
    
    check_default --> |不存在| create_default[创建默认分区<br/>（兜底）]
    check_default --> |存在| done[完成]
    create_default --> done

    style start fill:#e3f2fd
    style done fill:#c8e6c9
```

**触发时机**：
- 容器启动时（`entrypoint.sh`）
- 每次归档执行前（`ensure_current_partition`）

---

## 4. 归档流程

### 4.1 完整归档流程

```mermaid
flowchart TD
    subgraph 调度阶段["调度阶段"]
        A1[ArchiveScheduler<br/>等待调度时间] --> A2{到达执行时间?}
        A2 --> |是| A3[触发归档]
        A2 --> |否| A1
        A3 --> B1
    end

    subgraph 准备阶段["准备阶段"]
        B1[ensure_current_partition<br/>确保分区存在] --> B2[查询所有分区]
        B2 --> B3[过滤过期分区]
        B3 --> B4{有过期分区?}
        B4 --> |否| END1[归档完成<br/>无数据需处理]
        B4 --> |是| C1
    end

    subgraph 执行阶段["执行阶段（逐分区）"]
        C1[解析分区名<br/>获取月份范围] --> C2{分区上界<br/>≤ 阈值?}
        C2 --> |否| C3[跳过：活跃分区]
        C2 --> |是| C4{分区为空?}
        C4 --> |是| C5[DETACH + DROP<br/>空分区]
        C4 --> |否| C6[导出数据到<br/>JSONL+GZIP]
        C6 --> C7[更新元数据表<br/>archive_location]
        C7 --> C8[DETACH + DROP<br/>分区]
        C5 --> C9{还有分区?}
        C3 --> C9
        C8 --> C9
        C9 --> |是| C1
        C9 --> |否| END2[归档完成]
    end

    style A3 fill:#ffcdd2
    style C6 fill:#fff9c4
    style C7 fill:#e1f5fe
    style C8 fill:#c8e6c9
    style END2 fill:#c8e6c9
```

### 4.2 归档条件判断（关键逻辑）

```mermaid
flowchart TD
    subgraph 判断逻辑["过期判断逻辑"]
        P[分区: 2026_03] --> |解析| R1[范围: 3/1 ~ 4/1]
        R1 --> R2[分区上界: 4/1]
        
        T[配置: retention_days=30] --> |计算| TH[阈值: now - 30天<br/>= 4/13 - 30 = 3/14]
        
        R2 --> CMP{上界 ≤ 阈值?<br/>4/1 ≤ 3/14?}
        TH --> CMP
        
        CMP --> |否| SKIP[跳过：仍在活跃期]
        CMP --> |是| ARCHIVE[归档]
    end

    style CMP fill:#fff9c4
    style ARCHIVE fill:#c8e6c9
```

**判断公式**：
```
partition_end_date ≤ (now - retention_days) → 可归档
```

---

## 5. 保护机制

### 5.1 多层保护体系

```mermaid
graph TB
    subgraph 保护层["归档保护机制"]
        direction TB
        
        L1["第1层：时间保护"]
        L2["第2层：范围保护"]
        L3["第3层：数据保护"]
        L4["第4层：事务保护"]
        L5["第5层：试运行保护"]
    end

    L1 --> D1["✓ 整个分区都在阈值之前<br/>✓ 不归档部分月份数据"]
    L2 --> D2["✓ 只归档详情表<br/>✓ 元数据表永久保留"]
    L3 --> D3["✓ 先导出后删除<br/>✓ 文件验证通过才删除"]
    L4 --> D4["✓ 导出+更新+删除在同一事务<br/>✓ 失败自动回滚"]
    L5 --> D5["✓ dry-run 模式<br/>✓ 只导出不删除"]

    style L1 fill:#e3f2fd
    style L2 fill:#e8f5e9
    style L3 fill:#fff3e0
    style L4 fill:#fce4ec
    style L5 fill:#f3e5f5
```

### 5.2 保护机制详解

#### 第1层：时间保护

```python
# 只有整个分区都在阈值之前才归档
threshold = datetime.now() - timedelta(days=retention_days)
month_end = parse_partition_end(partition_name)

if month_end > threshold:
    # 分区仍在活跃期，跳过
    continue
```

**保护效果**：避免部分归档导致的数据割裂

#### 第2层：范围保护

```python
# 元数据表永不删除，只更新 archive_location 字段
UPDATE request_metadata
SET archive_location = '2026_03.jsonl.gz',
    archived_at = NOW()
WHERE unique_id IN (...)
```

**保护效果**：统计查询能力永久保留

#### 第3层：数据保护

```python
# 先导出验证，后删除分区
records = await cur.fetchall()  # 导出数据
write_gzip_file(records)        # 写入文件
verify_file()                   # 验证文件
# 验证通过后才执行删除
await detach_and_drop_partition()
```

**保护效果**：数据先安全导出，再删除分区

#### 第4层：事务保护

```python
async with conn.transaction():
    # 1. 更新元数据
    await conn.execute("UPDATE request_metadata SET archive_location = ...")
    # 2. 删除分区
    await conn.execute("ALTER TABLE ... DETACH PARTITION ...")
    await conn.execute("DROP TABLE ...")
    # 任一步骤失败，整体回滚
```

**保护效果**：原子性操作，一致性保证

#### 第5层：试运行保护

```bash
# --dry-run 模式
python scripts/archive_records.py --dry-run --retention-days 30
```

**保护效果**：
- ✓ 导出数据到文件
- ✗ 不更新 archive_location
- ✗ 不删除分区

### 5.3 错误处理流程

```mermaid
flowchart TD
    E1[归档错误] --> E2{错误类型}
    
    E2 --> |分区解析失败| E3[记录警告日志<br/>跳过该分区]
    E2 --> |数据导出失败| E4[记录错误日志<br/>事务回滚<br/>分区保留]
    E2 --> |元数据更新失败| E5[事务回滚<br/>分区保留]
    E2 --> |分区删除失败| E6[事务回滚<br/>归档文件保留]
    
    E3 --> E7[继续处理下一分区]
    E4 --> E8[等待 5 分钟后重试]
    E5 --> E8
    E6 --> E8
    
    E8 --> E9{重试次数?}
    E9 --> |未超限| E10[重新执行归档]
    E9 --> |超限| E11[停止归档<br/>发送告警]

    style E1 fill:#ffcdd2
    style E11 fill:#ffcdd2
```

---

## 6. 调度器设计

### 6.1 调度架构

```mermaid
sequenceDiagram
    participant App as 应用启动
    participant Scheduler as ArchiveScheduler
    participant Croniter as croniter
    participant Archiver as 归档执行器
    participant DB as 数据库

    App->>Scheduler: 初始化（读取配置）
    App->>Scheduler: start()
    Scheduler->>Croniter: 解析 cron 表达式
    Croniter-->>Scheduler: 下次执行时间
    
    loop 主循环
        Scheduler->>Scheduler: 计算等待时间
        Scheduler->>Scheduler: 心跳日志（每小时）
        Scheduler->>Archiver: 执行归档
        Archiver->>DB: 归档操作
        DB-->>Archiver: 结果
        Archiver-->>Scheduler: 归档统计
        Scheduler->>Croniter: 获取下次执行时间
    end
```

### 6.2 配置项

```yaml
archive:
  enabled: false                  # 启用开关
  retention_days: 30              # 保留天数
  storage_path: "/data/archives" # 存储路径
  schedule: "0 2 * * *"           # 每天凌晨2点
  timezone: "Asia/Shanghai"       # 时区
```

### 6.3 状态监控

```python
def get_status() -> Dict:
    return {
        "running": True,
        "schedule": "0 2 * * *",
        "last_run": "2026-04-13T02:00:00",
        "total_runs": 15,
        "total_records_archived": 12345,
        "last_result": {...}
    }
```

---

## 7. 归档文件格式

### 7.1 文件结构

```
/data/archives/
├── 2026_01.jsonl.gz    # 1月归档
├── 2026_02.jsonl.gz    # 2月归档
└── 2026_03.jsonl.gz    # 3月归档
```

### 7.2 数据格式

```jsonl
{"unique_id": "sess_001,req_001", "messages": [...], "created_at": "2026-03-15T10:30:00", ...}
{"unique_id": "sess_001,req_002", "messages": [...], "created_at": "2026-03-15T10:31:00", ...}
```

**格式优势**：
- JSONL：每行独立 JSON，支持流式读取
- GZIP：压缩率 ~10:1，节省存储空间

---

## 8. 运维指南

### 8.1 启用归档

```yaml
# config.yaml
archive:
  enabled: true
  retention_days: 30
  schedule: "0 2 * * *"
  timezone: "Asia/Shanghai"
```

### 8.2 手动归档

```bash
# 正常归档
python scripts/archive_records.py --retention-days 30

# 试运行（推荐先执行）
python scripts/archive_records.py --dry-run --retention-days 30
```

### 8.3 监控指标

| 指标 | 检查方法 | 告警阈值 |
|------|----------|----------|
| 调度器状态 | 日志含 "ArchiveScheduler 已启动" | 未启动 |
| 归档执行 | 日志含 "归档任务完成" | 连续失败 3 次 |
| 磁盘空间 | `df /data/archives` | 使用率 > 80% |
| 分区数量 | SQL 查询 pg_class | > 24 个分区 |

### 8.4 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 归档未执行 | `enabled: false` | 修改配置并重启 |
| 分区未删除 | 分区仍在活跃期 | 确认 retention_days 设置 |
| 磁盘空间不足 | 归档文件过多 | 扩容或迁移历史文件 |
| 查询报错 | JOIN 到已归档数据 | 检查 `archive_location IS NULL` 条件 |

---

## 9. 关键文件索引

| 文件 | 职责 |
|------|------|
| `traj_proxy/archive/scheduler.py` | 调度器实现 |
| `traj_proxy/archive/archiver.py` | 执行器实现 |
| `scripts/archive_records.py` | 独立归档脚本 |
| `traj_proxy/utils/config.py` | 配置加载 |
| `tests/e2e/layers/archive/` | E2E 测试用例 |
