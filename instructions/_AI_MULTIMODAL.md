**核心目的**：搞懂 **TTS、文生图、Sora** 背后的技术。这与 LLM 的 Next Token Prediction 不太一样，涉及 Diffusion 和 DiT。
# 👁️ 多模态与生成式 AI 协作指南

本文档专用于编写 **视觉生成、语音合成、视频模型、图文理解** 相关文档。
**Claude Code 需关注“模态对齐 (Alignment)”和“生成机制 (Generation)”。**

## 标准内容模板

### (Frontmatter)
```yaml
---
tech: [Flux / Sora / Whisper]
modality: [Text-to-Image / Video / Audio]
arch: [Diffusion / DiT / ViT]
---
````

### 1. 🎨 原理直觉 (Intuition)

> [!SUMMARY] 技术卡片
> 
> - **定义**：一种基于 [Diffusion/Transformer] 的 [模态] 生成模型。
> - **核心魔法**：它如何把文字变成像素/声波？(例如：DiT 将图像切块 Token 化，然后用 Transformer 预测噪声)。

### 2. 🏗️ 核心架构 (Architecture)

- **编码器 (Encoder)**：如何“看懂”文字？(如 T5 / CLIP)。
- **生成器 (Generator)**：如何“画出”图像？(如 UNet vs DiT)。
- **解码器 (Decoder)**：如何“还原”像素？(如 VAE)。

**数据流图 (Mermaid)**：

```mermaid
graph LR
    Text[提示词] -->|CLIP| Embedding
    Noise[随机噪声] -->|Denoise| Latent
    Embedding + Latent -->|DiT Block| NewLatent
    NewLatent -->|VAE Decode| Image
```

### 3. 🎬 关键技术点

- **对于视频**：如何保持时间连续性？(3D Attention)。
- **对于音频**：如何处理韵律和情感？(Flow Matching)。

### 4. 🚀 落地应用场景

- **Sora** -> 影视 Demo 预演。
- **ComfyUI** -> 工作流自动化绘图。