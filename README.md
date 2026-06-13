# MaskReview

MaskReview 是一个用于自动传播视频 mask 的复核工作台。它不是再做一个通用视频标注工具，而是专做 SAM2 自动传播之后的复核闭环。

核心场景是：用户先在第一帧给一个框，SAM2 把 mask 传播到整段视频；系统只把低置信或疑似漂移的帧放进复核队列，让用户用最少的点/框修正，再用 `交互次数 / 视频分钟` 衡量人工成本。

这个项目可以作为 CVAT 插件发布，也可以先做成轻量本地工具。对标 SAMannot 的自动标注能力，但补上它缺少的传播后复核交互层。

## 只做哪一环

已饱和的部分暂时不作为项目主卖点：

- 通用视频上传、播放和 mask 展示；
- 第一帧点/框提示；
- SAM2 mask propagation；
- 完整标注平台、任务管理、团队协作；
- 文本到框、多模型检索、视频编辑特效。

本项目收窄到一条闭环：

1. 第一帧给框，启动 SAM2 传播。
2. 用面积突变、空 mask、中心点跳变、贴边、帧间 mask IoU 等规则发现疑似漂移帧。
3. 生成低置信帧队列 `review_queue.json`。
4. 每个队列帧只请求一次最小修正：一个正/负点或一个更紧的框。
5. 记录 `estimated_min_interactions` 和 `estimated_interactions_per_video_minute`。
6. 后续接入从修正帧重新传播，比较修正前后的质量与人工成本。

## 当前 Demo

第一版是轻量本地/云端 GPU demo：

1. 上传短视频。
2. 输入第一帧目标框，格式为 `x1,y1,x2,y2`。
3. SAM2 生成并传播目标 mask。
4. 输出 overlay 视频、逐帧 mask、低置信帧队列和复核成本指标。

输出目录：

```text
outputs/<run_id>/
  frames/
  masks/
  overlay.mp4
  review_queue.json
  metrics.json
  corrections.json
  masks_after/
  overlay_after.mp4
  review_queue_after.json
  metrics_after.json
  comparison.json
```

`review_queue.json` 的每个 item 代表一个需要人工复核的帧，包含：

- `frame_index`
- `reason`
- `reasons`
- `frame_path`
- `mask_path`
- `recommended_correction`
- `estimated_interactions`
- `review_score`：0–1 的统一复核优先级（几何信号 + SAM2 置信度 + 合并状态），可用来给队列排序
- `status`：`needs_review` 或 `low_confidence_empty`（被消失段合并的帧）
- `diagnostics.confidence`：该帧 SAM2 置信度

`metrics.json` 里重点看：

- `review_frame_indices`
- `review_reason_counts`
- `mask_confidences`：逐帧 SAM2 置信度（峰值 object logit 的 sigmoid，~1.0 = 自信跟住，<0.5 = 疑似丢失/不在场）
- `estimated_min_interactions`
- `video_duration_seconds`
- `estimated_interactions_per_video_minute`
- `review_policy`

### 复核信号：几何 + SAM2 置信度

复核队列的触发原因（`reason`/`reasons`）除了几何信号（空 mask、面积突变/连降、中心跳变、帧间 IoU 下跌、贴边），还加入了 **`low_mask_confidence`**——SAM2 自身对该帧不确定（置信度低于 `ReviewPolicy.low_confidence_threshold`，默认 0.5）。每个队列项的 `diagnostics.confidence` 给出该帧置信度，便于分诊。两类信号**互补**：置信度抓「目标丢失/消失」（消失帧置信度≈0），几何抓「SAM2 自信但跟错相似目标」（此时置信度仍高）。

目标消失时会出现一长串「空 mask + 低置信」帧。`ReviewPolicy.collapse_low_confidence_runs`（默认开）把这种连续段合并成**一次 onset 交互**（其余帧 `estimated_interactions=0`、`status=low_confidence_empty`），帧仍保留在队列里可见——这样一次目标离场不会把人工成本估计撑大（实测某段消失视频 54 → 5 次交互）。

保存人工修正后，`corrections.json` 会记录每个复核帧的修正：

- `frame_index`
- `type`: `positive_point`、`negative_point` 或 `tight_box`
- `points`（可多个点，难帧常需要好几下）或 `box_xyxy`
- `created_at`
- `estimated_interactions`：该修正的**真实动作数**（点修正 = 点击数；框修正 = 1）

> 交互成本按真实动作计：点修正用 `;` 分隔多个点（如 `120,180;200,90` = 2 次交互），`actual_interactions` 汇总所有修正的真实动作数，替代「每帧算 1 次」的粗估。
>
> Gradio 里可以**直接点中心帧**来下点（多点连点）；Tight box 模式点两个对角即可。点击会自动填入下方坐标框（仍可手改）。

点击重传播后，系统会输出 `overlay_after.mp4`、`metrics_after.json` 和 `comparison.json`，用于对比修正前后的队列数量、人工交互次数和 mask 面积变化。

## Evaluation harness

批量评估用 manifest 驱动。可以先复制 `data/eval_manifest.example.json` 为 `data/eval_manifest.json`，也可以先把真实短视频放进 `data/eval_videos/`，再生成待填写的模板。

### Manifest template

真实视频放进 `data/eval_videos/` 后，先跑：

```bash
python scripts/generate_eval_manifest_template.py \
  --video-dir data/eval_videos \
  --output data/eval_manifest.template.json \
  --first-frame-dir data/eval_first_frames \
  --overwrite
```

脚本会扫描 `.mp4`、`.mov`、`.avi`、`.mkv`、`.webm` 等视频，导出每段视频的第一帧到 `data/eval_first_frames/`，并写出 `data/eval_manifest.template.json`。然后手动补齐每个 case 的：

- `init_box_xyxy`：第一帧目标框，格式 `[x1, y1, x2, y2]`。
- `object_name`：目标名称。
- `expected_challenge`：`clean_tracking`、`occlusion`、`similar_object_drift` 等分组标签。
- `expected_review_frames`：可选，人工认为应该进入复核队列的帧，例如 `[12, 18, 19]` 或 `"12,18-19"`。不确定时保留 `null`。

补完后保存为 `data/eval_manifest.json`，再跑评估：

```bash
python scripts/run_evaluation.py data/eval_manifest.json \
  --output-dir outputs/evaluation \
  --fixed-interval 10 \
  --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt
```

输出：

```text
outputs/evaluation/
  evaluation_results.csv
  evaluation_report.md
  <case_id>/
    overlay.mp4
    metrics.json
    corrections.json
    overlay_after.mp4
    metrics_after.json
    comparison.json
```

`evaluation_results.csv` 会把 MaskReview queue 和固定间隔复核 baseline 放在同一行里比较，重点看 `queue_reason_counts`、`queue_estimated_interactions`、`fixed_interval_review_frames` 和 `saved_interactions_vs_fixed_interval`。

如果 manifest 里填写了 `expected_review_frames`，报告还会输出：

- `queue_precision` / `queue_recall` / `queue_f1`：review queue 对人工标签的命中质量。
- `queue_missed_expected_frames`：人工认为该复核但队列漏掉的帧数。
- `queue_false_positives`：队列叫人复核但人工标签里没有的帧数。
- `fixed_interval_precision` / `fixed_interval_recall` / `fixed_interval_f1`：固定间隔 baseline 的同口径对照。

### Threshold calibration

评估时可以显式指定低置信队列阈值：

```bash
python scripts/run_evaluation.py data/eval_manifest.json \
  --output-dir outputs/evaluation-default \
  --area-jump-ratio 0.6 \
  --center-jump-ratio 0.25 \
  --iou-drop-threshold 0.2 \
  --area-decline-ratio 0.35 \
  --edge-margin 0
```

也可以一次跑三档校准策略：

```bash
python scripts/run_evaluation.py data/eval_manifest.json \
  --output-dir outputs/threshold-calibration \
  --fixed-interval 10 \
  --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --calibrate
```

输出：

```text
outputs/threshold-calibration/
  threshold_calibration.csv
  threshold_calibration_report.md
  sensitive/
  default/
  conservative/
```

三档策略含义：

- `sensitive`：少漏检，可能多叫人复核。
- `default`：当前默认平衡点。
- `conservative`：少叫人复核，可能漏掉更多漂移。

有 `expected_review_frames` 标签时，阈值选择优先看 `queue_recall` 和 `queue_missed_expected_frames`，再看 `saved_interactions_vs_fixed_interval`。没有标签时，校准只能说明复核量差异，不能证明漏检率。

### Ground-truth 质量评分（IoU / J&F）

如果某个 case 有逐帧真值 mask，可以在 manifest 里给它加一个 `ground_truth_mask_dir`，指向一个存放 `000000.png`、`000001.png`…（与预测 mask 同样的 6 位命名）的目录：

```json
{
  "id": "cup_01",
  "video_path": "eval_videos/cup_01.mp4",
  "init_box_xyxy": [120, 80, 240, 200],
  "ground_truth_mask_dir": "eval_gt/cup_01",
  "corrections": [ { "frame_index": 35, "type": "tight_box", "box_xyxy": [130, 60, 250, 210] } ]
}
```

评估时系统会用真值 mask 给传播结果打分（`src/quality_metrics.py`）：

- `J`：逐帧 region IoU（DAVIS 的 J）。
- `F`：边界 F-measure（DAVIS 的 F 近似，边界匹配容差为画面对角线的 0.8%）。
- `J&F`：两者均值。

`evaluation_results.csv` 会多出 `gt_jf_before`、`gt_jf_after`、`gt_jf_delta` 等列，`evaluation_report.md` 增加一节 **Ground-Truth Quality**，阈值校准报告增加 **Mean J&F b/a** 列。这样就能验证"修正不只是让队列变短，而是真的把 mask 质量修回来了"。没有 `ground_truth_mask_dir` 的 case 这些列留空，行为与之前一致。

## 运行方式

建议使用 Linux + CUDA + Python 3.10 或更新版本。SAM2 官方仓库要求先安装 PyTorch/TorchVision，再安装 SAM2。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/download_sam2_checkpoint.py --model tiny
python src/app.py --share --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt
```

如果云端平台已经预装 PyTorch/CUDA，可以从第二行开始。

### 命令行参数

- `--share`：创建 Gradio 公网链接。
- `--sam2-checkpoint`：SAM2 checkpoint 路径。
- `--sam2-model-cfg`：SAM2 config，默认 `configs/sam2.1/sam2.1_hiera_t.yaml`。
- `--device`：默认 `cuda`。
- `--vos-optimized`：启用 SAM2 最新版本支持的视频优化。

## MVP 验证问题

MVP 不证明“我也会做标注工具”，只证明这一点：

> SAM2 自动传播之后，系统能不能把人只叫到最该修的少数帧，并用更少交互恢复可用 mask？

最低实验设计：

| 方法 | 人工交互策略 | 主要指标 |
| --- | --- | --- |
| SAM2 baseline | 只给第一帧框 | J&F、失败帧数 |
| 固定间隔复核 | 每 N 帧人工检查 | J&F、交互次数/视频分钟 |
| Ours: review queue | 只检查低置信队列 | J&F、交互次数/视频分钟、队列命中率 |

核心指标：

- `interactions_per_video_minute`：每分钟视频需要多少次人工修正。
- `review_queue_precision`：队列里的帧有多少确实需要修。
- `post_review_quality`：修正后 J&F 或 IoU 是否恢复。
- `saved_interactions`：相比固定间隔复核少叫了多少次人。

## 实验结果（首批真值评测）

设置：SAM2.1 hiera **tiny**。**DAVIS-2017 480p** 单目标序列（GT = `mask>0`）；**MOSE train** 拥挤多目标场景（只跟其中一个实例，GT = `annotation==k`）；外加一段自采手机视频（388 帧，无真值）。J&F = region IoU（J）与 boundary F 的均值，对逐帧真值 mask 计算。修正 = 在漂移帧（IoU&lt;0.5）用真值框做一次 tight box，从修正帧重传播。GPU：RTX 3060，约 0.35–0.5s/帧（CPU 的 ~12–16×）。

**按数据集汇总**

| 数据集 | 序列数 | 平均 baseline J&F | 真失败帧 | 队列行为 |
| --- | ---: | ---: | ---: | --- |
| DAVIS-2017（单目标短片） | 5 | **0.96** | 0 | 几乎全程沉默；偶有几何误报（共 6 帧），0 漏检 |
| MOSE（拥挤/遮挡/相似目标） | 15 | **0.88**（方差大） | 多段 | 在真漂移上活跃；在「目标消失/重现」上大批误报 |

**修正回升 J&F（有真漂移的 MOSE 序列）**

| 序列 | 难点 | baseline J&F | 修正后 J&F | Δ |
| --- | --- | ---: | ---: | ---: |
| 23b9a3ea | 14 目标拥挤 | 0.787 | **0.924** | **+0.137** |
| 398d41f5 | 鸽群相似目标 | 0.590 | **0.676** | **+0.086** |
| fbfb6e30 | 拥挤+尺度 | 0.823 | 0.863 | +0.040 |
| 1fb378d9 | 40× 尺度变化 | 0.946 | 0.956 | +0.011 |
| 255f86ef | 拥挤+尺度 | 0.818 | 0.793 | −0.024 |
| fe470104 | 拥挤 | 0.863 | 0.852 | −0.011 |

**主要发现**

- **干净基准上 SAM2-tiny 很稳，队列正确沉默**：DAVIS 5 段 J&F 0.93–0.98，逐帧 IoU 从不跌破 0.5；队列只报了零星几帧，且 GT 证实全是误报（尺度变化/贴边触发，mask 实际无误），0 漏检。
- **队列的价值在 hard 真实场景**：相似目标的拥挤场景（鸟群 398d41f5、14 目标 23b9a3ea）SAM2-tiny 会把 mask 糊到邻近物体上甚至跟丢，队列精准命中，**1–2 个真值框修正即把 J&F 拉回**（最高 +0.137）。
- **诚实的边界**：修正并非总有效（6 例中 2 例轻微变差 −0.01~−0.02）——给一个部分正确的轨迹重新播种有时反而更糟；且在**目标消失/重现**时（6da17988、d262b7e4）几何队列大批误报（各 54 帧）而 SAM2 的 mask 其实正确。
- **长 messy 视频**：自采手机片全 388 帧，队列报 **72 帧**，准确圈出 **4 段以上跟丢episode**（含尾部 ~25 帧目标永久丢失），放过约 316 帧稳定帧。
- **结论**：几何启发式队列在干净/稳定视频上够用，但要在 hard 场景既抓真漂移又少误报（尤其消失重现），**需要把队列从几何代理升级到 SAM2 置信度/语义信号**——这是当前最该做的下一步。

> 口径与局限：仅用 tiny 模型；修正框由真值自动生成（演示「若给出正确框能恢复多少」，非真实人工标注成本）；MOSE 为多目标数据，本实验每段只跟一个实例。

## 发布形态

短期：轻量 Gradio 本地工具。

- 最快验证队列质量和交互指标。
- 适合放进 demo 视频、简历项目和实验报告。

中期：CVAT 插件。

- 读取已有视频任务和第一帧标注。
- 跑 SAM2 propagation。
- 在 CVAT 里生成 review queue。
- 用户只处理低置信帧。
- 导出修正后的 annotation 和交互统计。

## 当前文件结构

```text
maskreview/
  README.md
  PLAN_TRACKER.md
  docs/
    research_plan.md
    landing_readiness.md
  data/
    eval_manifest.example.json
    eval_first_frames/
    eval_videos/
  src/
    app.py
    corrections.py
    evaluation.py
    manifest_template.py
    pipeline.py
    sam2_runner.py
    video_io.py
  scripts/
    download_sam2_checkpoint.py
    generate_eval_manifest_template.py
    run_evaluation.py
    prepare_sample_video.py
  tests/
    test_app.py
    test_corrections.py
    test_evaluation.py
    test_manifest_template.py
    test_pipeline.py
  outputs/
```
