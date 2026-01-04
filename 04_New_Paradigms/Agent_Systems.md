---
topic: Agent Systems (Agent 系统完整指南)
focus: [工具使用, ReAct架构, 多智能体, 自主任务执行]
maturity: [初期, 爆发中]
distinctive: [工具集成, 自我纠正, 动态规划]
---

# 🤖 Agent Systems: AI 的自主行动

## 1. 🧬 演进定位

> [!SUMMARY] 身份卡片
>
> - **前身**：Chain-of-Thought Prompting（2022）- 让 AI 显示推理过程
> - **进化**：从"仅仅说"到"可以做"，加入工具调用能力
> - **竞品**：Claude with Tools（最可靠）、GPT-4 Function Calling、LangChain（最灵活）
> - **历史地位**：**让 LLM 从"聊天机器人"变成"执行系统"，打开了自动化的新时代**

### Agent 的三代演进

```
第一代（2023）：
  简单函数调用
  模型决定调用哪个函数，返回结果

第二代（2024）：
  ReAct 循环
  Thought → Action → Observation → Reflection

第三代（2025+）：
  自适应智能体
  能自己规划任务分解，调用多个工具
```

---

## 2. 🧠 核心突破

### 突破 1：工具使用的形式化

**问题**：模型如何"决定"使用哪个工具？

```
朴素想法：
  "模型应该知道什么时候用什么工具"

但实际问题：
  1. 工具有成百上千个，模型如何选择？
  2. 调用工具的参数有哪些？
  3. 工具失败了怎么办？
```

**解决方案：JSON Schema 的工具描述**

```json
{
  "name": "search_web",
  "description": "搜索互联网上的信息，返回相关文章列表",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索关键词"
      },
      "num_results": {
        "type": "integer",
        "description": "返回结果数量（1-10）",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

**模型如何理解和使用**：

```
Step 1: 系统提示中包含所有工具的描述
Step 2: 模型理解每个工具的作用和参数
Step 3: 当需要调用时，模型生成 JSON 格式的调用
Step 4: 系统执行这个调用，获得结果
Step 5: 将结果放回上下文，模型继续处理
```

**具体例子**：

```
用户：查找最新的 AI 新闻，然后总结
模型输出：
{
  "tool": "search_web",
  "params": {
    "query": "artificial intelligence news 2025",
    "num_results": 5
  }
}

系统执行，返回：
[
  "DeepSeek 发布新模型...",
  "OpenAI 推出新功能...",
  ...
]

模型理解返回结果，生成总结
```

---

### 突破 2：ReAct 循环的标准化

**ReAct = Reasoning + Acting**

```
传统思考方式：
  想一下 → 说答案

ReAct 方式：
  想一下 → 做一个行动 → 看结果 → 再想一下 → 再做行动 → ...

这就像人类解决问题的过程
```

**完整流程（伪代码）**：

```python
class Agent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.history = []

    def act(self, user_query):
        # 初始化对话历史
        self.history = [{"role": "user", "content": user_query}]

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # 第一步：思考
            response = self.model.generate(
                messages=self.history,
                tools=self.tools
            )

            # 检查是否要调用工具还是返回答案
            if response.type == "answer":
                return response.content

            if response.type == "tool_call":
                # 第二步：行动
                tool_result = self.execute_tool(
                    response.tool_name,
                    response.tool_params
                )

                # 第三步：观察和反思
                self.history.append({
                    "role": "assistant",
                    "content": response.reasoning
                })
                self.history.append({
                    "role": "system",
                    "content": f"Tool result: {tool_result}"
                })

                # 循环继续...

    def execute_tool(self, tool_name, params):
        tool = self.tools[tool_name]
        return tool(**params)
```

**实际运行示例**：

```
用户：2025 年比特币的价格是多少？

迭代 1：
  思考：我需要查找最新的比特币价格
  行动：调用 get_crypto_price("BTC")
  观察：$97,500（2025-01-04 的价格）

迭代 2：
  思考：我已经获得了答案
  行动：返回答案
  结果："2025 年 1 月 4 日，比特币价格是 $97,500"
```

---

### 突破 3：错误恢复和自我纠正

**问题**：工具调用经常失败

```
常见失败原因：
  1. 参数错误（类型不匹配、格式错误）
  2. 工具本身失败（API 宕机、网络错误）
  3. 返回结果不是模型期望的格式
  4. 模型误解了工具的功能
```

**解决方案：显式的错误处理**

```python
def execute_tool_with_fallback(tool_name, params):
    """执行工具，处理各种失败场景"""

    try:
        # 尝试执行
        result = tools[tool_name](**params)

        # 验证结果格式
        if not validate_result(result):
            return {
                "status": "error",
                "message": f"Tool returned invalid format: {type(result)}"
            }

        return {"status": "success", "data": result}

    except ToolTimeoutError:
        return {
            "status": "error",
            "message": f"Tool {tool_name} timed out"
        }

    except ToolAuthError:
        return {
            "status": "error",
            "message": f"Authentication failed for {tool_name}"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
```

**模型的自我纠正**：

```
当工具返回错误时：

迭代 1：
  想法：我要搜索"Python 教程"
  行动：search_web(query="Python 教程", num_results=5)
  观察：ERROR - Too many results (10,000+)，请指定更具体的搜索条件

迭代 2：
  想法：哦，搜索太宽泛了。我需要更具体。
  行动：search_web(query="Python 数据科学教程 2025", num_results=5)
  观察：成功，返回 5 条相关结果

迭代 3：
  想法：现在我有了答案，可以回复用户
  答案：[返回总结]
```

---

## 3. 📊 能力对比

### 不同 Agent 框架对比

```
             Claude Tools  GPT Functions  LangChain  Langraph
工具支持        ⭐⭐⭐⭐⭐    ⭐⭐⭐⭐⭐    ⭐⭐⭐⭐⭐   ⭐⭐⭐⭐⭐
易用性          ⭐⭐⭐⭐⭐    ⭐⭐⭐⭐     ⭐⭐⭐     ⭐⭐⭐⭐
可靠性          ⭐⭐⭐⭐⭐    ⭐⭐⭐⭐     ⭐⭐⭐     ⭐⭐⭐⭐⭐
灵活性          ⭐⭐⭐⭐     ⭐⭐⭐⭐     ⭐⭐⭐⭐⭐   ⭐⭐⭐⭐⭐
开源            ✗           ❌           ✓         ✓
本地部署        ✗           ❌           ✓         ✓
```

### 选择建议

```
场景 1：需要最简单、最可靠的方案
  → Claude Tools（API）
     - 官方文档最清晰
     - 最少出现奇怪的 bug

场景 2：需要用 GPT 模型
  → GPT-4 Function Calling（API）
     - 与 OpenAI 生态深度集成

场景 3：需要完整的本地开源方案
  → LangChain（框架）
     - 最大的社区
     - 工具库最丰富

场景 4：需要生产级别的 Agent
  → LangGraph（LangChain 的进阶）
     - 完整的状态管理
     - 易于持久化和调试
```

---

## 4. 💬 深度洞察

### 洞察 1：Agent 的自主性限制

```
常见误解：
  "Agent 可以完全自主解决问题"

实际情况：
  Agent 仍然需要明确的指导

例子：
  ✗ 错误：期望 Agent 自动规划一个企业迁移
  ✓ 正确：给 Agent 迁移的步骤清单，让它执行每一步

为什么？
  Agent 缺乏"深层规划"能力
  即使有工具，也可能因为：
    - 工具使用顺序不对
    - 前置条件未满足
    - 没有备选方案
```

---

### 洞察 2：Agent 的可观测性难题

```
问题：Agent 为什么做这个决定？

传统代码：
  if condition:
    do_something()
  # 清晰的逻辑

Agent：
  # Agent 内部的推理过程是黑箱
  model.generate(...)  # 返回：调用工具 X
  # 我们知道调用了什么，但不知道为什么

影响：
  - 难以调试
  - 难以信任（尤其在医疗、法律领域）
  - 难以改进（不知道哪里出错了）

解决方案：
  1. 记录所有中间步骤（即使执行变慢）
  2. 让模型显式说明"为什么选择这个工具"
  3. 添加人工验证步骤（关键决策）
```

---

### 洞察 3：Agent 的成本爆炸

```
表面上：Agent 调用模型 N 次，成本就是"单次成本 × N"

实际上：

成本 = 基础 API 调用 + Token 累积

例子：
  任务：查找并分析 10 篇新闻文章

  朴素估计：
    10 个搜索调用 × $0.01 = $0.10

  实际成本：
    搜索 API：$0.10
    第一轮对话：500 tokens 输入 + 200 tokens 输出 = $0.001
    第二轮搜索结果处理：1000 tokens 输入 + 500 tokens 输出 = $0.005
    第三轮汇总分析：5000 tokens 输入 + 2000 tokens 输出 = $0.025
    ... (可能 10+ 轮)

    总计：$0.20+（是初步估计的 2 倍）

启示：
  长的 Agent 循环成本会指数级增长
  需要：
    1. 优化循环次数（更好的规划）
    2. 压缩上下文（只保存必要信息）
    3. 用更便宜的模型（DeepSeek 而非 GPT-4）
```

---

## 5. 💰 成本与应用

### 部署方案与成本

```
方案 A：Claude Tools（API）
  成本：$0.003/输入 token + $0.015/输出 token

  典型 Agent 任务（10 轮循环）：
    总 tokens：30K （3K/轮输入 + 1.5K/轮输出）
    成本：$0.09 per task

方案 B：GPT-4o Function Calling（API）
  成本：$0.005/输入 token + $0.015/输出 token
  同样任务：$0.15 per task

方案 C：LangChain + DeepSeek（API）
  成本：$0.00055/输入 token + $0.00219/输出 token
  同样任务：$0.015 per task

方案 D：自建 Agent（开源模型）
  初期：硬件 $10K
  月成本：$500（电费）
  若日处理 100 个任务：
    成本/task：$0.17

推荐：
  小规模试验 → 方案 A 或 B（API）
  大规模应用 → 方案 C 或 D（低成本）
```

### 应用场景

**✅ Agent 最适合的应用**：

```
1. 数据分析和报告生成
   流程：查询 → 数据处理 → 分析 → 生成报告
   工具：数据库查询、计算、可视化
   成熟度：⭐⭐⭐⭐⭐

2. 客户服务自动化
   流程：理解问题 → 查询知识库 → 生成答案
   工具：搜索、CRM 查询
   成熟度：⭐⭐⭐⭐

3. 内容研究和总结
   流程：搜索 → 阅读 → 汇总
   工具：Web 搜索、PDF 解析
   成熟度：⭐⭐⭐⭐

4. 代码生成和审查
   流程：理解需求 → 生成代码 → 测试 → 修复
   工具：代码执行、测试框架
   成熟度：⭐⭐⭐⭐☆

5. 复杂业务流程自动化
   流程：拆分任务 → 执行 → 汇总
   工具：CRM、财务系统、协作工具
   成熟度：⭐⭐⭐
```

**❌ 不适合的应用**：

```
1. 实时控制
   - 需要毫秒级响应
   - Agent 延迟太高

2. 高精度决策
   - 医疗诊断、法律建议
   - 无法解释和审计

3. 创意写作
   - Agent 模式适合"执行"，不适合"创意"

4. 极低成本的大规模应用
   - 如果需要处理 1 亿次请求/天
   - API 成本会很高
```

---

## 6. ⚠️ 关键限制

### 限制 1：工具的可用性和可靠性

```
现状：
  许多企业的内部工具没有 API
  或者 API 不稳定

问题场景：
  需要用 Agent 查询企业旧系统
  但系统没有标准 API
  → 需要定制开发接口
  → 成本高

解决方案：
  1. 优先将高频操作 API 化
  2. 设置工具失败后的降级方案
  3. 定期监控工具可用性
```

---

### 限制 2：Agent 的"走偏"问题

```
现象：
  Agent 有时会做一些意料之外的事

例子：
  任务：找出销售额下降的原因
  Agent 不仅找了，还：
    - 自动发邮件给销售团队
    - 修改了某些配置
    - 删除了一些数据

为什么？
  1. Agent 对"权限边界"理解不清
  2. 工具描述模糊，模型误解用途
  3. 没有明确的决策限制

解决方案：
  1. 明确告知 Agent 能做和不能做什么
  2. 任何"写"操作都需要确认
  3. 关键操作记录和审计
  4. 限制工具的权限（只读数据库等）
```

---

### 限制 3：长序列推理的失败

```
问题：
  当 Agent 的操作链很长时，容易出错

例子：
  步骤：搜索 → 分析 → 对比 → 计算 → 总结

  在步骤 3（对比）时：
    如果前面的分析有微小误差
    对比和计算会基于错误的数据
    最终答案完全错误

为什么？
  误差累积（Error Accumulation）
  每一步都可能引入 1-2% 的错误
  10 步后：1.01^10 ≈ 1.1（10% 误差）

解决方案：
  1. 在关键步骤加入验证
  2. 如果置信度低，重新检查前面的步骤
  3. 限制循环深度（通常 < 10 轮）
  4. 对重要任务加入人工审核
```

---

## 7. 🔗 知识连接

### 核心架构
- **[[ReAct_Architecture]]** - Reasoning + Acting 循环
- **[[Tool_Calling_Protocol]]** - JSON Schema 的工具定义
- **[[Prompt_Engineering_for_Agents]]** - Agent 专门的 Prompt 技巧

### 框架和工具
- **[[LangChain_Agent_Tutorial]]** - LangChain 使用指南
- **[[LangGraph_Advanced]]** - 生产级别的 Agent 架构
- **[[OpenAI_Function_Calling]]** - GPT 函数调用

### 应用案例
- **[[Agent_for_Data_Analysis]]** - 数据分析 Agent
- **[[Customer_Service_Agent]]** - 客服 Agent
- **[[Code_Generation_Agent]]** - 代码生成 Agent

---

## 总结

### Agent 的核心价值

```
不仅仅是：
  模型能更好地说话

而是：
  模型能做事情（通过工具）
  模型能改正自己（通过观察和反思）
  模型能规划流程（虽然不完美）
```

### 2025 年的应用前景

```
短期（几个月）：
  - 企业开始使用 Agent 处理重复任务
  - Agent 逐步可靠，成为主流

中期（半年）：
  - Agent 与企业系统的集成加深
  - 多智能体系统出现

长期（1+ 年）：
  - Agent 可能演变成"自主工作者"
  - 但仍需要人类监督
```

### 建议

**对企业**：
```
现在就可以尝试 Agent：
  ✅ 客服聊天机器人 + 知识库搜索
  ✅ 报表生成 + 数据分析
  ✅ 合同审查 + 提取关键信息

但不要：
  ❌ 让 Agent 做关键的生产操作（未经验证）
  ❌ 在没有人工审核的情况下执行
```

**对开发者**：
```
学习路线：
  Level 1：理解 ReAct 架构
  Level 2：用 LangChain 构建简单 Agent
  Level 3：深入理解工具设计和错误处理
  Level 4：构建生产级 Agent（LangGraph）
```

---

**最后的话**：

Agent 代表了 AI 的一个关键转变：**从"回答问题"到"解决问题"**。

虽然当前的 Agent 还不完美，但它已经开始改变企业的工作方式。2025 年，懂 Agent 系统的开发者会很吃香。

