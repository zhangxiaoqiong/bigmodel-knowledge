---
topic: Transformer (Transformer 架构)
category: Architecture
difficulty: ⭐⭐⭐⭐⭐
tags: [基础, 架构, 必修, 核心]
---

# 🏗️ Transformer Architecture

## 1. 📖 本质定义 (First Principles)

> [!SUMMARY] 概念卡片
>
> - **定义**：Transformer 是一种纯粹基于**自注意力机制**（Self-Attention）的深度学习架构，用于处理序列数据。它由编码器（Encoder）和解码器（Decoder）两个堆叠的相同模块组成，每个模块包含多头注意力层和前馈网络层。
>
> - **历史背景**：在 Transformer 之前，RNN/LSTM 是序列处理的标准方法，但它们有两个致命弱点：(1) **顺序处理**导致无法并行化，训练速度慢；(2) **长距离梯度**衰减，难以学习长文本中的依赖关系。Transformer 通过自注意力机制打破了顺序约束，使得所有位置可以并行计算。
>
> - **核心直觉**：与其让一个"隐状态"像传送带一样逐步传递信息，不如让每个位置都能直接"查询"其他所有位置。这就是自注意力的核心。

### 🎯 设计目标

| 目标 | 解决的问题 | Transformer 方案 |
|------|-----------|-----------------|
| **并行化** | RNN 必须顺序计算，无法 GPU 加速 | 所有位置同时计算注意力，O(log n) 深度 |
| **长依赖** | LSTM 无法有效捕捉长距离关系 | 每个位置都能直接与所有位置交互 |
| **可扩展性** | 难以处理超长文本 | 线性堆叠层数，模型容量可控 |

---

## 2. 📐 整体架构 (The Math)

### 2.1 编码器-解码器框架

```
输入序列：[token_1, token_2, ..., token_n]
            |
            v
        ┌─────────────────────────┐
        │   编码器 (Encoder)      │  ← 理解输入
        │  ┌─ Multi-Head Att      │
        │  ├─ Feed Forward        │
        │  └─ (重复 N 次)         │
        └─────────────────────────┘
            |
            v
        ┌─────────────────────────┐
        │  解码器 (Decoder)       │  ← 生成输出
        │  ┌─ Masked Multi-Head   │
        │  ├─ Cross-Attention     │
        │  ├─ Feed Forward        │
        │  └─ (重复 N 次)         │
        └─────────────────────────┘
            |
            v
        输出序列：[token_1, token_2, ..., token_m]
```

### 2.2 单层（Transformer Block）的细节

#### 编码器层 (Encoder Block)

```
输入 x_l: (batch, seq_len, d_model)
    |
    v
┌──────────────────────────────┐
│ 多头自注意力层                 │
│ x_l' = MultiHeadAttention(    │
│        x_l, x_l, x_l) + x_l   │← 残差连接
└──────────────────────────────┘
    |
    v
┌──────────────────────────────┐
│ 层归一化 (LayerNorm)          │
│ y = LayerNorm(x_l')           │
└──────────────────────────────┘
    |
    v
┌──────────────────────────────┐
│ 前馈网络 (Feed-Forward)       │
│ FFN(y) = ReLU(yW_1 + b_1)W_2  │
│          + b_2 + y            │← 残差连接
└──────────────────────────────┘
    |
    v
输出 x_{l+1}: (batch, seq_len, d_model)
```

#### 数学表达

$$\text{Encoder}(x_l) = \text{FFN}(\text{LayerNorm}(\text{MultiHeadAtt}(x_l) + x_l)) + \text{MultiHeadAtt}(x_l)$$

**注意**：论文中的顺序是 LayerNorm 在前（Pre-Norm），也有改进版是在后（Post-Norm）。

#### 关键概念说明

| 概念 | 作用 | 为什么需要它 |
|------|------|-----------|
| **残差连接** (Residual) | $\text{output} = f(x) + x$ | 防止深层网络梯度消失，允许梯度直通 |
| **层归一化** (LayerNorm) | 将 $x$ 归一化为均值 0、方差 1 | 稳定训练，加速收敛 |
| **前馈网络** | 两层全连接：$d_{model} \to d_{ff} \to d_{model}$ | 引入非线性，通常 $d_{ff} \approx 4 \times d_{model}$ |

### 2.3 前馈网络的细节

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**为什么用两层？**

- 第一层：$d_{model} \to d_{ff}$（通常 $d_{ff} = 4 \times d_{model}$，即 2048）
  - 目的：**投影到高维空间**，增加表达能力
  - 引入 ReLU 非线性

- 第二层：$d_{ff} \to d_{model}$（降维回原始维度）
  - 目的：**投影回原始空间**，确保与后续层兼容

**参数量计算**：
$$\text{Parameters}_{FFN} = d_{model} \times d_{ff} + d_{ff} \times d_{model}$$
$$= 2 \times d_{model} \times d_{ff} = 2 \times d_{model} \times 4 \times d_{model} = 8 d_{model}^2$$

对于 $d_{model} = 768$（BERT 大小）：
$$= 8 \times 768^2 \approx 4.7 \text{ 百万参数}$$

前馈网络通常占 Transformer 总参数的 **60-70%**。

### 2.4 解码器层的特殊之处

解码器在编码器基础上增加了一个**交叉注意力**层：

```
┌─────────────────────────────────────┐
│ 1. 自注意力 (Masked)                 │  ← 只能看已生成的 token
│    Att(x, x, x, mask=causal)        │
└─────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────┐
│ 2. 交叉注意力 (Cross-Attention)     │  ← 查询编码器输出
│    Att(decoder, encoder, encoder)   │
│    Q 来自解码器，K、V 来自编码器    │
└─────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────┐
│ 3. 前馈网络                          │
└─────────────────────────────────────┘
```

---

## 3. 🔬 原理实现 (NumPy from Scratch)

### 完整 Transformer 实现

```python
import numpy as np
from typing import Optional, Tuple

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax"""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """层归一化"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

class PositionalEncoding:
    """位置编码：为每个位置添加独特的信号"""

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len).reshape(-1, 1)  # (max_seq_len, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = np.sin(position * div_term)  # 偶数位：sin
        pe[:, 1::2] = np.cos(position * div_term)  # 奇数位：cos

        self.pe = pe

    def __call__(self, seq_len: int) -> np.ndarray:
        """返回前 seq_len 的位置编码"""
        return self.pe[:seq_len, :]

class MultiHeadAttention:
    """多头注意力"""

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 初始化投影矩阵（实际应该通过训练学习）
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单头注意力"""
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)  # (batch, seq_len, d_v)

        return output, attention_weights

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        参数：
            Q, K, V: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len) 或 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.shape[0]

        # 线性投影并分割为多个头
        Q = np.matmul(Q, self.W_q).reshape(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
        K = np.matmul(K, self.W_k).reshape(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)
        V = np.matmul(V, self.W_v).reshape(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        # 对每个头计算注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )  # (batch, num_heads, seq_len, d_k)

        # 连接所有头
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.d_model
        )  # (batch, seq_len, d_model)

        # 最终线性投影
        output = np.matmul(attn_output, self.W_o)

        return output, attention_weights

class FeedForwardNetwork:
    """前馈网络"""

    def __init__(self, d_model: int, d_ff: int):
        self.W_1 = np.random.randn(d_model, d_ff) * 0.01
        self.b_1 = np.zeros((1, d_ff))
        self.W_2 = np.random.randn(d_ff, d_model) * 0.01
        self.b_2 = np.zeros((1, d_model))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        """
        hidden = np.maximum(0, np.matmul(x, self.W_1) + self.b_1)  # ReLU
        output = np.matmul(hidden, self.W_2) + self.b_2
        return output

class TransformerEncoderBlock:
    """Transformer 编码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def __call__(
        self,
        x: np.ndarray,
        self_attn_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        x: (batch_size, seq_len, d_model)
        """
        # 多头注意力 + 残差
        attn_output, _ = self.attention(x, x, x, self_attn_mask)
        x = x + attn_output

        # 层归一化
        x = layer_norm(x)

        # 前馈网络 + 残差
        ffn_output = self.ffn(x)
        x = x + ffn_output

        # 层归一化
        x = layer_norm(x)

        return x

class TransformerEncoder:
    """Transformer 编码器（多层堆叠）"""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int
    ):
        self.layers = [
            TransformerEncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        self.pos_encoding = PositionalEncoding(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]

        # 加上位置编码
        pos_enc = self.pos_encoding(seq_len)  # (seq_len, d_model)
        x = x + pos_enc[np.newaxis, :, :]  # 广播到 batch

        # 通过所有层
        for layer in self.layers:
            x = layer(x)

        return x

# 测试示例
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads, num_layers, d_ff = 8, 2, 256

    # 创建随机输入
    X = np.random.randn(batch_size, seq_len, d_model)

    # 创建 Encoder
    encoder = TransformerEncoder(d_model, num_layers, num_heads, d_ff)

    # 前向传播
    output = encoder(X)

    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
```

### 关键实现细节

1. **位置编码（Positional Encoding）**：
   - 使用正弦和余弦函数
   - 不同频率的波形编码不同的位置
   - 优点：可外推到更长的序列

2. **残差连接（Residual Connection）**：
   - $x_{l+1} = \text{SubLayer}(x_l) + x_l$
   - 好处：梯度可以直接反向传播

3. **层归一化（Layer Normalization）**：
   - 对最后一维（特征维）进行归一化
   - 稳定训练，加速收敛

---

## 4. 🧠 关键机制剖析 (Deep Dive)

### Q1: 为什么需要位置编码？

**问题**：Self-Attention 是**排列不变**的。

```python
x = [token_1, token_2, token_3]
x_shuffled = [token_3, token_1, token_2]

# 虽然顺序改变，但自注意力的计算结果是一样的！
# 因为注意力只关心"哪些 token 与哪些 token 相似"，
# 不关心它们在序列中的位置
```

**解决方案**：直接把位置信息编码到输入中。

$$\text{Positional Encoding}(pos, 2i) = \sin(pos / 10000^{2i/d_{model}})$$
$$\text{Positional Encoding}(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}})$$

**为什么用三角函数？**

- 对任意固定距离 $\delta$，$PE_{pos+\delta}$ 可以表示为 $PE_{pos}$ 的线性函数
- 模型可以学会相对位置关系
- 可以外推到更长的序列（这是相对位置编码的优势）

### Q2: 为什么前馈网络有 4 倍的隐层？

$$d_{ff} = 4 \times d_{model}$$

这没有严格的数学推导，而是经验上发现的：

- 太小：表达能力不足
- 太大（如 8 倍）：参数过多，训练困难，效果反而下降
- 4 倍：最佳平衡点

### Q3: 为什么用多头注意力而不是单头的大注意力矩阵？

```
单头：(d_model, d_model) 矩阵，参数 = d_model²
     但矩阵秩有限，难以捕捉多种关系

多头：h 个 (d_k, d_k) 矩阵，总参数 = h × d_k² = d_model²
     但不同的头学到不同的投影空间

类比：用多个不同焦距的镜头观察世界，vs 用一个镜头
```

实验证明，多头注意力比单头性能好得多。

### Q4: 残差连接的作用是什么？

```
没有残差连接：
  x_0 ---> Layer_1 ---> Layer_2 ---> ... ---> Layer_N

梯度反向传播：
  ∂L/∂x_0 = ∂L/∂x_N × ∂x_N/∂x_{N-1} × ... × ∂x_1/∂x_0

如果每个 ∂x_i/∂x_{i-1} < 1，梯度呈指数衰减（梯度消失）！

有残差连接：
  x_1 = f(x_0) + x_0

梯度：
  ∂x_1/∂x_0 = ∂f/∂x_0 + 1
             ≥ 1（即使 ∂f/∂x_0 = 0，梯度也能通过 "+1" 项流动）
```

这是为什么现在可以训练几百层的深度网络。

### 面试考点

1. **"Transformer 的时间复杂度是多少？"**
   - 多头注意力：O(L² × d_{model})
   - 前馈网络：O(L × d_{model}²)
   - 总体：O(L² × d_{model}) （注意力是瓶颈）

2. **"为什么 Transformer 比 RNN 更易并行化？"**
   - RNN：第 t 步依赖第 t-1 步，必须顺序计算
   - Transformer：所有位置的注意力可以同时计算

3. **"Attention 和 Recurrence 哪个更适合捕捉长距离依赖？"**
   - Attention：直接连接，距离为 1 步
   - Recurrence：距离为 t 步，需要梯度经过多个时间步

4. **"解码器为什么需要 masked self-attention？"**
   - 生成时，第 t 个 token 不能看到第 t+1 个 token
   - 否则就"作弊"了（偷看答案）

---

## 5. 🔗 知识连接

### 核心组件

- **[[Attention_Mechanism]]** - Transformer 的灵魂
- **[[Position_Encoding]]** - 位置信息的编码方式
- **[[Layer_Normalization]]** - 训练稳定性保障

### 衍生架构

- **[[BERT]]** - 仅编码器的 Transformer（双向）
- **[[GPT]]** - 仅解码器的 Transformer（自回归）
- **[[Vision_Transformer_ViT]]** - 将图像分块后用 Transformer 处理

### 现代优化

- **[[Flash_Attention]]** - 降低注意力计算复杂度
- **[[Sparse_Attention]]** - 选择性注意力
- **[[Grouped_Query_Attention_GQA]]** - 减少推理内存
- **[[MLA]]** - DeepSeek 使用的改进注意力

---

## 📊 性能对比

| 模型 | 架构 | 训练速度 | 长文本能力 | 推理成本 |
|------|------|---------|---------|--------|
| LSTM | Recurrent | 慢 ⭐ | 弱 ⭐ | 中等 |
| Transformer (原版) | Self-Attention | 快 ⭐⭐⭐⭐⭐ | 强 ⭐⭐⭐⭐ | 高 ⭐⭐⭐⭐⭐ |
| Flash Attention | Optimized Attention | 快 ⭐⭐⭐⭐⭐ | 强 ⭐⭐⭐⭐ | 低 ⭐⭐⭐⭐ |

---

## 📚 推荐资源

1. **原始论文**：《Attention Is All You Need》(Vaswani et al., 2017)
2. **详细讲解**：The Illustrated Transformer (Jay Alammar)
3. **实现参考**：Annotated Transformer (Alexander Rush)

---

**最后的直觉**：

Transformer 是深度学习的一次范式转移。它用**注意力替代循环**，用**并行替代顺序**。这使得模型可以同时看到整个序列，并且可以深度叠加而不损失梯度。这个简单而优雅的设计成为了所有现代大模型的基石。
