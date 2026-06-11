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
2. 用面积突变、空 mask、置信度信号等规则发现疑似漂移帧。
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
```

`review_queue.json` 的每个 item 代表一个需要人工复核的帧，包含：

- `frame_index`
- `reason`
- `frame_path`
- `mask_path`
- `recommended_correction`
- `estimated_interactions`

`metrics.json` 里重点看：

- `review_frame_indices`
- `estimated_min_interactions`
- `video_duration_seconds`
- `estimated_interactions_per_video_minute`
- `review_policy`

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
promptable-video-segmentation/
  README.md
  docs/
    research_plan.md
  src/
    app.py
    pipeline.py
    sam2_runner.py
    video_io.py
  scripts/
    download_sam2_checkpoint.py
    prepare_sample_video.py
  tests/
    test_pipeline.py
  outputs/
```
