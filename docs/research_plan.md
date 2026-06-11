# 研究计划：MaskReview

## 一句话题目

MaskReview：自动标注先跑完，人只复核低置信帧，用最少交互修正 SAM2 传播漂移，并用 `交互次数 / 视频分钟` 衡量人工成本。

## 项目边界

这不是一个新的通用标注工具，也不是完整 CVAT 替代品。

项目只补一环：SAM2 已经把第一帧标注传播到全视频之后，哪些帧最值得叫人回来修。

暂缓范围：

- 文本提示和自然语言检索；
- 完整视频编辑产品；
- 多人协作标注平台；
- 从零实现 SAM2 或追求模型 benchmark SOTA；
- 大而全的 annotation UX。

必须完成的范围：

- 低置信帧队列；
- 每个队列帧的最少修正建议；
- 交互次数统计；
- `interactions_per_video_minute` 指标；
- 修正前后质量和人工成本对比。

## 对标对象

对标 SAMannot 的方向不是“我也能自动标注”，而是：

- SAMannot/SAM2 类工具覆盖自动传播；
- 本项目覆盖传播后的复核交互；
- 重点证明自动传播后不需要人逐帧扫，只需要处理系统挑出的少量风险帧。

## MVP 工作流

### Phase 1：本地 review pass

输入：

- 视频；
- 第一帧目标框；
- SAM2 checkpoint。

输出：

- `overlay.mp4`
- `masks/`
- `review_queue.json`
- `metrics.json`

当前低置信代理信号：

- 空 mask；
- mask 面积突变；
- 后续可加入 SAM2 logits、边界漂移、检测器重定位不一致。

每个 review queue item 必须给出：

- 帧号；
- 触发原因；
- 原帧路径；
- mask 路径；
- 推荐修正动作；
- 预计交互次数。

### Phase 2：最少交互修正

目标不是做完整标注 UI，而是实现三种最小 correction：

- 单个正点；
- 单个负点；
- 单个 tighter box。

修正策略：

1. 用户只打开 review queue 里的帧。
2. 每帧默认只允许一次 correction。
3. 从修正帧向后重新传播。
4. 记录修正次数和修正后 mask 质量。

### Phase 3：CVAT 插件化

插件职责：

- 从 CVAT task 读取视频和已有第一帧标注；
- 调用 SAM2 传播；
- 把低置信帧写回 CVAT review queue；
- 在 CVAT 内只让用户处理这些帧；
- 导出 annotation 和交互统计。

本地工具先证明算法与指标，CVAT 插件再证明工作流能接真实标注环境。

## 实验设计

### Baselines

| 方法 | 交互策略 | 说明 |
| --- | --- | --- |
| SAM2-only | 第一帧一次框 | 不做传播后复核 |
| Fixed review | 每 N 帧检查一次 | 常见但浪费人工 |
| Ours: low-confidence queue | 只检查队列帧 | 项目主张 |

### 指标

| 指标 | 意义 |
| --- | --- |
| J&F / IoU | mask 质量 |
| drift frame count | 明显传播失败帧数量 |
| review queue precision | 队列里真正需要修的比例 |
| review queue recall | 真正漂移帧被队列抓住的比例 |
| interactions per video minute | 人工成本主指标 |
| saved interactions vs fixed review | 相比固定间隔少叫了多少次人 |

### 数据

优先级：

1. 自采短视频 10 到 20 段：最容易展示真实漂移和复核价值。
2. DAVIS 2017 小样本：用于基本 J&F/IoU 对比。
3. YouTube-VOS 子集：用于更长视频和遮挡场景。

## 里程碑

### M0：已完成或正在完成

- 第一帧框输入；
- SAM2 propagation；
- overlay 和 masks 输出；
- 基于 mask 面积突变的 review frame detection；
- `review_queue.json`；
- `estimated_interactions_per_video_minute`。

### M1：可展示 MVP

- 在 Gradio 里展示低置信帧队列；
- 点击队列帧可看到原图和当前 mask；
- 支持单点或单框 correction；
- 从 correction frame 重新传播；
- 输出修正前后指标对比。

### M2：实验可信度

- 标注 10 到 20 段自采视频的 drift/failure 帧；
- 比较 SAM2-only、fixed review、low-confidence queue；
- 产出一张主表：质量、交互次数、每分钟交互。

### M3：CVAT 插件雏形

- CVAT task 读取；
- review queue 写回；
- correction 记录；
- annotation 导出；
- 交互统计导出。

## 简历/Agent 叙事

这个项目可以串成一条 agentic workflow：

> 自动标注先做完，自检模块找出低置信帧，只在必要时请求人类最小修正，然后把修正继续传播并计量节省了多少人工。

比“做一个标注工具”更强的点在于：

- 边界窄；
- 输入输出稳定；
- 有自检；
- 有 human-in-the-loop；
- 有成本指标；
- 能接 CVAT 这种真实工作流。
