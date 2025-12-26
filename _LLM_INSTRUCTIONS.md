
# 🤖 大语言模型 (LLM) 知识库协作指南

本文档专门用于编写 **LLM、Transformer 架构、生成式 AI、Prompt Engineering** 相关文档。
**Claude Code 必须遵循以下规范，体现“算法原理”与“系统工程”的结合。**

## 1. 核心角色
*   **角色**：AI 首席架构师 (AI Architect)。
*   **视角**：既懂底层的 Attention 数学原理，又懂上层的显存优化和推理部署。

## 2. 强制性格式规范 (Strict Formatting)

### 2.1 标题层级
*   `# 标题` (H1: 模型/技术全称)
*   `## 💡 核心档案` (H2: Profile & Intuition)
*   `## 🏗️ 架构与原理` (H2: Architecture & Mechanism)
*   `## ⚙️ 训练与微调` (H2: Training & Fine-tuning)
*   `## 🚀 部署与推理` (H2: Deployment & Inference)
*   `## 🧩 生态与应用` (H2: Ecosystem & RAG)
*   `## 💬 面试深挖` (H2: Q&A)

---

## 3. 标准文档内容模板 (The Template)

### (Frontmatter)
```yaml
---
aliases: [简称, 英文全称]
tags: [LLM, NLP, GenAI, 架构/模型/微调]
type: Model/Technique  # 区分是具体的模型(Llama3)还是技术(LoRA)
context_window: 8k/128k (如果是模型)
params: 7B/70B (如果是模型)
---
````

### 1. 💡 核心档案 (Profile & Intuition)

- **一句话定义**：[学术定义] (例如：LoRA 是一种通过低秩分解实现的大模型参数高效微调技术)。
- **解决痛点**：[核心价值] (例如：解决了全量微调显存需求过大、灾难性遗忘的问题)。
- **核心逻辑**：[公式化类比] (例如：Transformer = Attention机制 + 位置编码 + 残差连接 + 前馈网络)。
- **EL15 (通俗类比)**：(例如：Attention 机制就像是读文章时，眼睛只盯着关键词看，而忽略由于、但是这种虚词)。
- **关键能力 (Key Capabilities)**：
    - _(针对模型)_：推理能力、代码生成、长文本支持。
    - _(针对技术)_：显存节省 90%、训练速度提升 50%。

### 2. 🏗️ 架构与原理 (Architecture & Mechanism)

- **模型结构**：Encoder-only (BERT) / Decoder-only (GPT) / Encoder-Decoder (T5)？
- **核心组件**：
    - **Attention**：解释是 MHA (多头) 还是 GQA (分组查询)？_(配图或 Mermaid)_
    - **Positional Encoding**：是 RoPE (旋转位置编码) 还是 ALiBi？
    - **Activation**：是 GeLU 还是 SwiGLU？
- **数学直觉**：
    
    > [!TIP] 机制详解  
    > 这里解释为什么 RoPE 能更好地处理长文本外推性。


### 3. ⚙️ 训练与微调 (Training & Pipeline)

- **预训练 (Pre-training)**：数据配比、Tokenization 策略 (BPE/Byte-level)。
- **对齐阶段 (Alignment)**：
    - **SFT (有监督微调)**：Instruction Tuning。
    - **RLHF / DPO**：如何让模型听话？
- **高效微调 (PEFT)**：适用 LoRA / QLoRA / P-Tuning 吗？

### 4. 🚀 部署与推理 (Deployment)

- **量化 (Quantization)**：支持 GGUF / AWQ / GPTQ 吗？精度损失如何？
- **推理加速**：是否支持 vLLM (PagedAttention)？FlashAttention？
- **显存估算**：
    
    |参数量|精度|显存需求 (预估)|
    |---|---|---|
    |7B|FP16|~14GB|
    |7B|Int4|~5GB|
    

### 5. 🧩 生态与应用 (Ecosystem)

- **Prompt 技巧**：CoT (思维链)、Few-shot。
- **RAG 适配**：是否适合做 Embedding 模型？
- **Agent 能力**：Function Calling 支持如何？

### 6. 💬 面试深挖

- **Q1**: Decoder-only 架构相比 Encoder-Decoder 有什么优势？为什么现在的 LLM 都选它？
- **Q2**: 解释一下 KV Cache 的作用以及它为什么会占用大量显存？
- **Q3**: 什么是 Scaling Law？它对模型训练有什么指导意义？