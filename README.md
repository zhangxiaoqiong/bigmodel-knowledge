### 📂 第一部分：Obsidian 终极目录结构 (The Architecture)

请在你的 Obsidian 仓库中建立以下 7 个核心文件夹。这是按照**技术分层**和**模态扩张**的逻辑组织的。

```text
📦 AI_Knowledge_Base
 ┣ 📂 00_Timeline_&_Roadmap (进化全景)
 ┃ ┣ 📜 GenAI_History_Map.md (从 Transformer 到 GPT-5 的时间轴)
 ┃ ┗ 📜 Tech_Radar_2025.md (当前技术雷达：哪些是炒作，哪些是落地)
 ┃
 ┣ 📂 01_Foundations (基石原理 - 不变的底层)
 ┃ ┣ 📜 Transformer.md
 ┃ ┣ 📜 Backpropagation.md
 ┃ ┗ 📜 Attention_Mechanism.md
 ┃
 ┣ 📂 02_Model_Lineage (模型族谱 - 核心大脑)
 ┃ ┣ 📂 GPT_Family (OpenAI路线: 3.5 -> 4 -> 4o -> o1)
 ┃ ┣ 📂 Llama_Family (开源基座路线)
 ┃ ┣ 📂 Sparse_Models (MoE路线: Mixtral, DeepSeek)
 ┃ ┗ 📂 Efficient_Models (端侧小模型: Phi, Gemma)
 ┃
 ┣ 📂 03_Multimodality (感官延伸 - 视觉/听觉/视频)
 ┃ ┣ 📂 Visual_Gen (Diffusion, Flux, Midjourney)
 ┃ ┣ 📂 Video_Gen (Sora, Kling, Runway)
 ┃ ┣ 📂 Audio_TTS (Whisper, CosyVoice)
 ┃ ┗ 📂 Vision_Language (ViT, Qwen-VL, GPT-4V)
 ┃
 ┣ 📂 04_New_Paradigms (新范式 - 思考与行动)
 ┃ ┣ 📂 Reasoning (推理模型: CoT, RL-Search, o1)
 ┃ ┗ 📂 Agents (智能体: Tool Use, Planning, Multi-Agent)
 ┃
 ┣ 📂 05_Engineering (工程落地 - RAG/Ops)
 ┃ ┗ (这里放之前的 RAG, Prompt, Eval 内容)
 ┃
 ┗ 📂 06_Products_&_Scenarios (应用场景 - 解决问题)
   ┣ 📜 AI_Coding (Cursor, Windsurf)
   ┣ 📜 AI_Search (Perplexity, SearchGPT)
   ┗ 📜 AI_Browser (Browser Use)
```

### 📂 第二部分：Claude 协作指令集 (The Prompt System)

为了填充这个庞大的架构，你需要 4 份不同的指令文件。请保存在 `instructions/` 目录下。

#### 1. `_AI_BASE.md` (负责 01 目录)

#### 2. `_AI_LINEAGE.md` (负责 02 目录 - 模型族谱)

#### 3. `_AI_MULTIMODAL.md` (负责 03 目录 - 全模态)

#### 4. `_AI_APP.md` (负责 04, 05, 06 目录 - 应用与新范式)

*(使用之前提供的 **Workflow/应用版** 指令即可，它非常适合写 RAG、Agent 和具体产品分析。只需增加一个“产品分析”章节)*


---

### 🚀 如何开始构建？(最佳实践路径)

现在你有了图纸（目录）和工具（提示文件），建议按以下顺序让 Claude 帮你填充内容，从而建立你的理解：
**第一步：梳理脉络 (00_Timeline) - 继续**

> **指令示例：**  
> `claude "读取 _AI_LINEAGE.md，帮我生成一份 'GenAI_History_Map.md'。我要从 2017年 Transformer 论文开始，经过 BERT, GPT-3, ChatGPT, Llama, 到现在的 DeepSeek V3 和 Sora。请用 Mermaid 时间轴列出关键节点，并简述每个节点的‘突变’意义。"`

- **你的收获**：你会得到一张清晰的 AI 进化地图，知道现在处于什么位置，未来可能去哪。

---

**第二步：攻克当前最热模型 (02_Model_Lineage)**  
_不要去学 GPT-3 了，那是考古。直接学 DeepSeek V3 和 GPT-4o。_

> **指令示例 1 (DeepSeek)：**  
> `claude "读取 _AI_LINEAGE.md 和 _AI_EDGE.md。帮我深度解析 'DeepSeek_V3.md'。重点解释它是如何用 MoE (Mixture-of-Experts) 和 MLA (Multi-Head Latent Attention) 实现高性能低成本的。对比它和 Llama 3.1 的架构区别。"`

> **指令示例 2 (o1/Reasoning)：**  
> `claude "读取 _AI_LINEAGE.md。帮我分析 OpenAI 的 'o1_Preview.md'。重点解释 'Chain of Thought (CoT)' 和 'Test-time Compute' (推理时计算) 的概念。为什么说它是大模型的新范式？"`

- **你的收获**：你将理解为什么现在大家都在谈论 MoE 和推理模型，以及“慢思考”为什么比“快回答”更重要。

---

**第三步：理解多模态魔法 (03_Multimodality)**  
_不仅是文字，我们要看图生视频。_

> **指令示例 (Sora/Video)：**  
> `claude "读取 _AI_MULTIMODAL.md。帮我写一篇关于 'Sora_and_VideoGen.md' 的文档。解释 'DiT (Diffusion Transformer)' 架构是如何结合了 Transformer 的扩展性和 Diffusion 的生成能力的。为什么它能理解物理规律？"`

> **指令示例 (Flux/Image)：**  
> `claude "读取 _AI_MULTIMODAL.md。帮我分析 'Flux_1.md'。它相比 Stable Diffusion XL 有什么改进？什么是 'Flow Matching'？"`

- **你的收获**：你将明白视频生成不是简单的“动起来的图片”，而是“世界模拟器”。

---

**第四步：掌握应用新范式 (04_New_Paradigms & 06_Products)**  
_这是你提到的“智能体”、“自动化编程”落地的地方。_

> **指令示例 (Agent)：**  
> `claude "读取 _AI_APP.md。帮我写一篇 'AI_Agents_Overview.md'。解释从简单的 Prompt 到 'ReAct' 模式，再到现在的 'Multi-Agent' (如 CrewAI) 的演进。画出 Agent 调用工具的 Mermaid 流程图。"`

> **指令示例 (Coding)：**  
> `claude "读取 _AI_APP.md。帮我分析 'AI_Coding_Tools.md' (以 Cursor 和 Windsurf 为例)。重点分析它们是如何利用 'Codebase Indexing' (代码库索引) 和 'Speculative Edits' (推测编辑) 来改变编程体验的。"`

- **你的收获**：你将看到 AI 是如何从“聊天机器人”变成“数字员工”和“编程搭档”的。

---

### 💡 总结：这个知识库如何帮你“跟踪前沿”？

有了这套系统，当**明天**又出了一个新模型（比如 GPT-5 发布了），你只需要做一件事：

1. **判断类别**：它是纯语言模型？还是多模态？还是新架构？
2. **选择指令**：如果是新架构，用 `_AI_LINEAGE.md`；如果是多模态，用 `_AI_MULTIMODAL.md`。
3. **生成文档**：让 Claude 帮你把新知识填进这个框架里。
4. **建立连接**：问 Claude：“GPT-5 相比目录里的 DeepSeek V3 和 GPT-4o，主要改进在哪里？”

这样，你的知识库就是**活的**，它会随着技术的发展而生长，而不是变成一堆过时的 PDF。