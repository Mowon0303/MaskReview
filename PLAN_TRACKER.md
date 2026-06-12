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
| P2 | Review queue | 把疑似漂移帧转成可操作队列 | `review_queue.json`, `build_review_queue()`，面积/空 mask/中心跳变/贴边/帧间 IoU 信号 | Done | 单元测试覆盖面积突变、空 mask、中心跳变、贴边、IoU 下跌和输出路径 | 用真实视频校准阈值，并后续接入 SAM2 logits |
| P3 | 成本指标 | 记录最少人工交互成本 | `estimated_min_interactions`, `estimated_interactions_per_video_minute` | Done | `tests/test_pipeline.py` 验证指标写入 result 和 metrics | 后续用真实人工修正次数替代 estimate |
| P4 | Demo UI | 在轻量本地工具中展示队列和 KPI | `src/app.py` Gradio 页面，队列帧预览面板 | Done | 页面能展示 overlay、review KPI、low-confidence queue，并能选择队列帧查看原帧/mask | 继续优化队列命中信号和操作体验 |
| P5 | 最少交互修正 | 每个队列帧只做一次点/框 correction | `src/corrections.py`, `corrections.json`, Gradio correction 表单 | Done | 支持正点、负点、tight box 三种 correction，并写入 run artifact | 将保存的 correction 接入重传播 |
| P6 | 修正后重传播 | 从 correction frame 继续传播并比较前后效果 | `segment_with_corrections()`, `overlay_after.mp4`, `metrics_after.json`, `comparison.json` | Done | correction prompt 接入 SAM2 后能输出 before/after metrics | 继续验证多 correction、多漂移视频和质量指标 |
| P7 | 实验评估 | 证明队列比固定间隔复核更省交互 | `src/evaluation.py`, `src/manifest_template.py`, `scripts/run_evaluation.py`, `scripts/generate_eval_manifest_template.py`, `data/eval_manifest.example.json`, `data/eval_first_frames/`, `threshold_calibration_report.md` | Partial | 已能从 manifest 生成 eval CSV/report，支持真实视频模板生成、第一帧导出、`expected_review_frames` 命中率/漏检率统计和 sensitive/default/conservative 阈值校准；真实视频证据和 IoU/J&F 未完成 | 准备 10 到 20 段自采视频，补 `init_box_xyxy` 和 `expected_review_frames`，并跑第一版校准 report |
| P8 | CVAT 插件雏形 | 把本地复核闭环接入真实标注工作流 | 未实现 | Pending | 从 CVAT task 读取、写回 review queue、导出 annotation 和交互统计 | 本地 MVP 稳定后再开插件目录 |

## 下一步执行清单

1. 准备 10 到 20 段真实短视频，用固定间隔复核和 review queue 复核做对比。
2. 运行 `scripts/generate_eval_manifest_template.py --video-dir data/eval_videos --output data/eval_manifest.template.json --first-frame-dir data/eval_first_frames --overwrite` 生成模板和第一帧。
3. 补齐 `init_box_xyxy`、`object_name`、`expected_challenge`；能人工判断漂移位置的 case 同时补 `expected_review_frames`。
4. 保存为 `data/eval_manifest.json`，运行 `scripts/run_evaluation.py data/eval_manifest.json --calibrate`，比较 sensitive/default/conservative 三档阈值的复核量、命中率和漏检率。
5. 记录真实人工交互次数，逐步替代 `estimated_min_interactions`。
6. 增加 correction 后质量回升验证：先用 mask 面积恢复、队列减少、人工修正帧数作为轻量指标，再接入 IoU/J&F。
7. 汇总失败样例，决定下一批 drift 信号阈值和 SAM2 logits 接入优先级。

## 不做清单

- 不做完整 CVAT 替代品。
- 不做文本提示和自然语言检索，除非复核闭环已经跑通。
- 不做视频编辑特效。
- 不把重点放在 SAM2 复现本身，重点是传播后的自检和 human-in-the-loop 复核。
