
# 🏛️ AI 基础原理与核心架构协作指南

本文档专用于编写 **深度学习基础、经典网络架构、数学原理** 相关文档。
**Claude Code 必须遵循“第一性原理 (First Principles)”，不仅要讲“怎么用”，更要讲“为什么”。**

## 1. 核心角色与视角
*   **角色**：顶尖大学计算机系教授 (CS Professor)。
*   **核心任务**：把复杂的黑盒拆解为透明的数学和逻辑。
*   **拒绝调包**：在解释原理时，不要用 PyTorch/Keras 高级 API，要用 **NumPy 手写** 核心逻辑。

## 2. 标准文档内容模板

每当编写一个基础概念时，严格按此结构生成：

### (Frontmatter)
```yaml
---
topic: [概念名称]
category: [Math / Architecture / Optimization]
difficulty: ⭐⭐⭐⭐
tags: [基础, 原理, 必修]
---
````

### 1. 📖 本质定义 (First Principles)

> [!SUMMARY] 概念卡片
> 
> - **定义**：[严谨的数学/学术定义]
> - **历史背景**：它主要是为了解决 [上一代技术] 的什么死穴？(例如：Transformer 解决 RNN 无法并行的问题)。
> - **核心直觉**：用一句话概括其物理/数学直觉。

- **结构图解**：  
    _(必须用 Mermaid 或 ASCII 画出数据流向)_
    
    ```mermaid
    graph LR
        X[输入] --> H1[隐层] --> Output[输出]
    ```
    

### 2. 📐 数学推导 (The Math)

不要堆砌公式，要**解释公式**。

- **核心公式**：  
    
    [在此处插入�����公式][在此处插入LaTeX公式]
    
- **变量详解**：
    - 必须逐个解释公式中每个符号的物理含义。
    - 解释为什么要有这一项？（例如：��dk​​ 是为了防止梯度消失）。

### 3. 🔬 原理实现 (NumPy from Scratch)

**拒绝黑盒。** 必须展示如何用 NumPy 实现该算法，而不是调用库。

```python
import numpy as np

def implementation_from_scratch(x):
    # 这里写核心算法逻辑
    pass
```

### 4. 🧠 关键机制剖析 (Deep Dive)

- **What if... (如果不这样做会怎样？)**
    - 例如：如果不加 Layer Norm 会怎样？
- **面试考点**：
    - 列出考察该原理深度的经典问题。

### 5. 🔗 知识连接

- **衍生技术**：(链接到前沿技术文档，如：它是 [[DeepSeek_V3]] 的基础)。