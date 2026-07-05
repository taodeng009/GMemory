# GMemory Atomic Merge Phase 1 实施计划

## 1. 背景

当前 GMemory merge 的主要问题是：合并后的 insight 更长、更综合、重复更少，但容易变成 checklist-like 规则。其直接原因包括：

- `limited_number` 对输出数量的压缩过于激进；
- prompt 强调合并和去重，但没有要求保留触发条件和策略原子性；
- 对 LLM 输出只做编号解析，没有长度和 checklist 质量检查。

Phase 1 的目标是以最小改动验证上述假设，提高 merged insight 的简洁性、条件性和原子性。

## 2. 核心约束

1. **不修改原 merge 逻辑。** 现有 `merge_insights()` / `_merge_rules()` 作为 `original` 基线保留。
2. 新机制以独立的 `atomic_v1` 策略实现，只有通过环境变量显式启用才生效。
3. 默认值保持 `original`，保证已有实验、持久化数据和运行行为不变。
4. Phase 1 不改造 insight JSON schema，不改检索、score、正负任务关联和任务聚类机制。

## 3. 配置设计

在 API 配置和 `.env` / `template.env` 中增加：

```env
# original: 使用原 GMemory merge，行为不变。
# atomic_v1: 使用 Phase 1 的原子化 merge。
GMEMORY_API_MERGE_STRATEGY=original

# atomic_v1 的目标保留比例；0.33 表示 10 条输入最多输出 4 条（向上取整）。
GMEMORY_API_ATOMIC_MERGE_RATIO=0.33

# atomic_v1 单条 insight 的最大英文单词数。
GMEMORY_API_ATOMIC_MERGE_MAX_WORDS=30
```

参数校验：

- `MERGE_STRATEGY` 只接受 `original` 和 `atomic_v1`，非法值回退到 `original`；
- ratio 必须在 `(0, 1]` 范围，非法值回退到 `0.33`；
- max words 必须为正整数，非法值回退到 `30`。

现有配置仍独立控制 merge 是否执行以及执行周期：

```env
GMEMORY_API_MERGE=enabled
GMEMORY_API_MERGE_STEPS=20
```

## 4. 代码结构

### 4.1 保留 original 路径

不改动现有 `_merge_rules(rules)` 的内部实现。它仍是原 GMemory 实验基线。

### 4.2 增加策略调度

在 `InsightsManager` 增加配置字段，由调度器选择：

```python
if merge_strategy == "atomic_v1":
    merged_rules = self._merge_rules_atomic_v1(related_rules)
else:
    merged_rules = self._merge_rules(related_rules)
```

`merge_insights()` 的任务聚类、入库和持久化流程在 Phase 1 中保持不变。

### 4.3 新增 `_merge_rules_atomic_v1()`

`atomic_v1` 仍使用每批最多 10 条输入，但输出上限改为：

```python
limited_number = max(1, math.ceil(len(batch) * merge_ratio))
```

ratio 默认为 `0.33`，输出上限使用向上取整：

| 输入数 | original 上限 | atomic_v1 上限 |
|---:|---:|---:|
| 10 | 1 | 4 |
| 9 | 1 | 3 |
| 6 | 0 | 2 |
| 3 | 0 | 1 |
| 1 | 0 | 1 |

## 5. Atomic Merge Prompt

新增以下独立 prompt，不覆盖 `merge_rules_system_prompt` 和 `merge_rules_user_prompt`。

### 5.1 System Prompt

```python
atomic_merge_rules_system_prompt = """
You are an agent skilled at summarizing and distilling insights. You are given a list of insights that were previously extracted from similar tasks. These insights may contain redundancy or overlap.

Your job is to **merge and consolidate genuinely similar insights**, and output a refined version that is **clear, actionable, concise, and atomic**.

NOTE:
- All merged insights **must be based strictly on the given inputs**. You are **not allowed to make up** or infer any new information.
- The output should be easy to read and follow.
- Each output insight should contain one main recommendation under one applicability condition.
- Merge insights only when their triggering conditions, recommended strategies, execution phases, preconditions, and failure modes are compatible.
- Preserve important conditions, exceptions, applicability boundaries, and causal relationships.
- Do not merge insights that apply to different execution phases.
- Do not merge insights that depend on different preconditions.
- Do not merge insights that address different failure modes.
- Do not combine multiple sequential actions into a checklist or an end-to-end procedure.
- When uncertain whether insights should be merged, keep them separate.
- Each insight must contain no more than {max_words} English words.

📑 Output Format:
- Start your response directly with the numbered list, no preamble or explanations.
- Each insight should be a short sentence.
- Use the following format exactly:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""
```

### 5.2 User Prompt

```python
atomic_merge_rules_user_prompt = """
## Here are the current insights that need to be merged:
{current_rules}

## Please consolidate and rewrite them into **no more than {limited_number} atomic insights**.

Each output insight must contain no more than {max_words} English words.

Do not force unrelated insights together to reach the output limit.

Your output:
"""
```

## 6. 输出质量门控

Phase 1 增加轻量级、可解释的后处理，不修改 original 路径。

### 6.1 检查项

对 `atomic_v1` 的每条输出检查：

1. 文本不能为空；
2. 单词数不超过 `max_words`；
3. 不得超过 prompt 允许的输出数量。

### 6.2 失败处理

为避免静默丢失 insight：

- 首次输出不合格时，使用审查反馈最多重试 1 次；
- 重试仍不合格时，保留本批原始 rules，而不是接受长 checklist 或返回空列表；
- 日志记录失败原因、重试次数和 fallback 结果。

## 7. 可观测性

在 `insights.log` 中为每批 merge 记录：

- merge strategy；
- 输入/输出 insight 数量；
- 实际压缩比；
- 每条输出的单词数；
- 是否发生 retry 或 fallback。

这些信息用于比较 `original` 与 `atomic_v1`，不影响 insight JSON schema。

## 8. 测试计划

### 8.1 配置测试

- 默认使用 `original`；
- `atomic_v1` 能通过 env 启用；
- strategy、ratio 和 max words 的非法值正确回退；
- 现有 `MERGE=disabled` 时，无论 strategy 为何都不执行 merge。

### 8.2 单元测试

- 输出上限按 ratio 正确计算，且不会为 0；
- original 路径仍调用原 `_merge_rules()`；
- atomic_v1 路径调用新 prompt；
- 超长、空输出和超出数量上限能被检测；
- 第一次失败会重试，第二次失败会 fallback 到原始 rules。

### 8.3 回归测试

- 在不设置新 env 的情况下，original 的 prompt、输出解析和持久化行为不变；
- 现有 insight 检索和 backward 反馈测试不受影响。

## 9. 实验评估

在相同 trajectory 集合和模型上运行：

```text
A: MERGE_STRATEGY=original
B: MERGE_STRATEGY=atomic_v1, RATIO=0.33, MAX_WORDS=30
C: MERGE=disabled
```

建议记录：

- merge 前后 insight 数量；
- 平均单词数和 P95 单词数；
- 每条 insight 的并列动作数；
- checklist-like insight 比例；
- 条件触发型 insight 比例；
- 语义重复率；
- 下游任务成功率和 token 成本。

Phase 1 成功标准：

1. checklist-like 比例明显下降；
2. merged insight 平均长度下降；
3. 规则语义重复率不高于 original；
4. 下游成功率不显著低于 original；
5. original 基线的行为完全保持。

## 10. 实施顺序

1. 增加配置项及 env 校验。
2. 将配置传递到 `InsightsManager`。
3. 增加策略调度，保留 original 分支。
4. 新增 atomic merge prompt。
5. 实现 `_merge_rules_atomic_v1()` 和 ratio 计算。
6. 实现非空、长度和输出数量检查、一次 retry 和 fallback。
7. 增加日志与单元测试。
8. 运行 original/atomic_v1/disabled 三组对照实验。

## 11. Phase 1 不包含的内容

以下改动放到后续阶段，避免 Phase 1 无法归因：

- 不按 insight embedding 再次聚类；
- 不将 insight 改造为 condition/action/reason 结构化 schema；
- 不修改 merged insight 的 score 或正负任务关联继承逻辑；
- 不改任务 FINCH 聚类方式；
- 不改 insight 检索排序和 backward 奖惩机制。
