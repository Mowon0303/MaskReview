# Plan Tracker

项目定位：MaskReview 是 SAM2 传播后的复核闭环。系统自动标注先跑完，只把低置信或疑似漂移帧放进队列，让人用最少点/框修正，并用 `interactions_per_video_minute` 衡量人工成本。

## 状态说明

| 状态 | 含义 |
| --- | --- |
| Done | 已实现并有本地验证 |
| Partial | 已有雏形，但还不是可展示闭环 |
| Next | 下一步优先做 |
| Pending | 排期内但暂不开始 |

## 完成表

| ID | 模块 | 目标 | 当前产物 | 状态 | 验收方式 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 项目定位 | 从通用标注工具收窄到 MaskReview: SAM2 propagation review loop | `README.md`, `docs/research_plan.md` | Done | 第一屏明确低置信帧队列、最少交互、CVAT 插件/本地工具形态 | 后续所有功能都围绕复核闭环，不扩成通用标注平台 |
| P1 | SAM2 baseline | 第一帧框输入后跑完整视频传播 | `src/sam2_runner.py`, `src/pipeline.py` | Done | 能输出 `overlay.mp4`, `masks/`, `metrics.json` | 在 GPU 环境跑真实 SAM2 sample |
| P2 | Review queue | 把疑似漂移帧转成可操作队列 | `review_queue.json`, `build_review_queue()` | Done | 单元测试覆盖面积突变、空 mask 队列项和输出路径 | 加入 SAM2 logits、边界漂移等更强低置信信号 |
| P3 | 成本指标 | 记录最少人工交互成本 | `estimated_min_interactions`, `estimated_interactions_per_video_minute` | Done | `tests/test_pipeline.py` 验证指标写入 result 和 metrics | 后续用真实人工修正次数替代 estimate |
| P4 | Demo UI | 在轻量本地工具中展示队列和 KPI | `src/app.py` Gradio 页面 | Partial | 页面能展示 overlay、review KPI、low-confidence queue | 点击队列帧查看原帧/mask，并提交 correction |
| P5 | 最少交互修正 | 每个队列帧只做一次点/框 correction | 未实现 | Next | 支持正点、负点、tight box 三种 correction | 设计 correction schema，并写入 `corrections.json` |
| P6 | 修正后重传播 | 从 correction frame 继续传播并比较前后效果 | 未实现 | Pending | 输出 before/after metrics | 接入 SAM2 `add_new_points_or_box` 的中间帧修正调用 |
| P7 | 实验评估 | 证明队列比固定间隔复核更省交互 | 未实现 | Pending | 表格包含 J&F/IoU、队列 precision/recall、交互次数/分钟 | 准备 10 到 20 段自采视频和 DAVIS 小样本 |
| P8 | CVAT 插件雏形 | 把本地复核闭环接入真实标注工作流 | 未实现 | Pending | 从 CVAT task 读取、写回 review queue、导出 annotation 和交互统计 | 本地 MVP 稳定后再开插件目录 |

## 下一步执行清单

1. 实现 `corrections.json` 数据结构：`frame_index`, `type`, `points`/`box`, `created_at`。
2. 在 Gradio 里支持选择一个 review queue frame，并显示原帧和当前 mask。
3. 支持单次 point/box correction 输入，先只保存 correction，不急着重传播。
4. 加一个 fake segmenter 测试，验证 correction 被记录到 run artifact。
5. 在 GPU 环境跑一个短视频，把 `overlay.mp4`, `review_queue.json`, `metrics.json` 作为第一份 demo 证据。

## 不做清单

- 不做完整 CVAT 替代品。
- 不做文本提示和自然语言检索，除非复核闭环已经跑通。
- 不做视频编辑特效。
- 不把重点放在 SAM2 复现本身，重点是传播后的自检和 human-in-the-loop 复核。
