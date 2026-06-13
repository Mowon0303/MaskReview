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
| P7 | 实验评估 | 证明队列比固定间隔复核更省交互 | `src/evaluation.py`, `src/quality_metrics.py`, `src/manifest_template.py`, `scripts/run_evaluation.py`, `scripts/generate_eval_manifest_template.py`, `data/eval_manifest.example.json`, `data/eval_first_frames/`, `threshold_calibration_report.md` | Partial | 已能从 manifest 生成 eval CSV/report，支持真实视频模板生成、第一帧导出、`expected_review_frames` 命中率/漏检率统计和 sensitive/default/conservative 阈值校准；首段真实视频 CPU smoke 已完成（见下方「P7 实验记录」），IoU/J&F 评分已实现（`src/quality_metrics.py`，给 manifest case 加 `ground_truth_mask_dir` 即可打分），多视频与真值标注证据未完成 | 准备 10 到 20 段自采视频，补 `init_box_xyxy` 和 `expected_review_frames`，并跑第一版校准 report |
| P8 | CVAT 插件雏形 | 把本地复核闭环接入真实标注工作流 | `docs/cvat_plugin.md`（设计）, `src/cvat/export.py`, `tests/test_cvat_export.py` | In progress | 设计文档完成；step 1 离线导出层已实现并单测（masks→COCO RLE、CVAT mask-shape RLE、review_queue→issues、interaction_report，10 测试）；step 2 活的 cvat-sdk client + CLI 待用户起本地 Docker CVAT 后再做 | 用户搭好本地 Docker CVAT（见 `docs/cvat_plugin.md` §4）后写 `src/cvat/client.py` + `scripts/cvat_sync.py`，对真实 task 联调 |

## P7 实验记录

### 2026-06-12 · 首段真实视频 CPU smoke（SAM2.1 hiera tiny）

环境：Windows，CPU-only（torch 2.5.1+cpu，无 GPU）。数据：`data/eval_videos/8589b8c5…mp4`（720×1280，29.9fps，388 帧），取前 80 帧做 CPU smoke。首帧框 `[360, 600, 625, 765]`。产物：`outputs/real-smoke-80/`。

- **Baseline 复核 pass**：80 帧 / 460.8s（5.76s/帧）。复核队列 **20/80** 帧 `[31–42, 56, 70–76]`；原因统计 `mask_area_trending_down×10, empty_mask×7, mask_area_spiked×4, mask_area_dropped×3, mask_touches_frame_edge×2`；`estimated_min_interactions=20`。SAM2 两次跟丢目标（mask 面积在 35–40、74 帧归零），队列**精准命中两段漂移、放过约 60 帧稳定帧**。
- **修正 → 重传播 pass**：在两个丢失帧各画 1 个 tight box 重新框住目标（`frame 35 / 74`），重传播 417s。复核队列 **20 → 10** `[21, 40, 56, 70–76]`；**空 mask 帧 7 → 0**；第一段漂移（31–42）完全恢复（仅余 1 个过渡帧），frame 35/74 的 mask 面积各回升约 3 万像素；`actual_interactions=2`。第二段（70–76）目标小且贴帧边，修正后 mask 仍偏移，队列**正确保留该段复核**（自检未被误清）。

**结论**：在该真实视频上，复核队列把人工从「逐帧扫 80 帧」压到 20 帧；2 次最小修正即消除全部空 mask 帧并把队列腰斩。
**口径/局限**：仍只用交互数 / 队列数与 mask 面积代理，**无 ground-truth IoU/J&F**；单视频、CPU tiny；首帧框为人工估计，跟踪目标（瓶子）与首帧种子区域不完全一致。单元测试 `python -m unittest discover -s tests` → 22/22 通过。

### 2026-06-12 · DAVIS 真值 J&F 评测（SAM2.1 hiera tiny，CPU）

用 `remotezip` 从 DAVIS-2017 480p 流式取单目标序列（只取需要的几十个小文件，数据全部落 E:），通过 `ground_truth_mask_dir` 接入新的 J&F 评分；首帧框由 GT bbox 自动给出。

| 序列 | 帧数 | J&F（真值） | 复核队列 | 真实失败帧(IoU<0.85) | 备注 |
| --- | ---: | ---: | --- | ---: | --- |
| blackswan | 50 | **0.9684**（J 0.957 / F 0.979） | 空(0) | 0 | 干净跟踪，队列正确「沉默」，比固定间隔少叫 4 次 |
| drift-straight | 50 | **0.9685**（J 0.958 / F 0.979） | [23,36,44,45] | 0 | 车大幅缩放+贴边触发 area_trending_down/touches_edge，但实际 IoU 0.96–0.98 → 4 帧全是误报 |
| car-roundabout | 75 | **0.9774**（J 0.986 / F 0.969） | 空(0) | 0 | 含树丛遮挡，tiny 仍稳跟，全程 IoU 0.98+，队列零误报 |
| parkour | 100 | **0.9645**（J 0.942 / F 0.987） | [23] | 0 | 快速人体运动+运动模糊，仍无丢失（IoU 全程≥0.90），队列 1 帧误报 |
| dance-twirl | 90 | **0.9300**（J 0.915 / F 0.945） | [35] | 0 | 旋转裙摆飞起，frame 33–43 IoU 跌到最低 0.77（最接近失败的一次）但未丢失，随后恢复 0.92+ |

**新评分解锁的关键收获**：有了真值 J&F，第一次能把「队列挑的帧」和「真正坏的帧」对账。

- blackswan：队列空，确无坏帧 → 零误报零漏检（队列在稳定视频上不过度打扰）。
- drift-straight：队列报了 4 帧，但 GT 显示这 4 帧 IoU 都 0.96+（见 `outputs/_davis_drift_pred_vs_gt.jpg`，绿=预测紧贴红=GT）→ **几何启发式在剧烈尺度变化/贴边时会误报**，不过本序列没有漏掉真失败（本序列无真失败）。这是 landing_readiness gap#1（队列仍是几何代理）的量化证据。
- SAM2 tiny 在这五段 DAVIS 上都跟得很好（J&F 0.93–0.98，含遮挡的 car-roundabout、快速的 parkour、旋转自遮挡的 dance-twirl 都无丢失；全集逐帧 IoU 最低也只到 0.77，没有任何一帧跌破 0.5），所以这几轮没有「修正回升 J&F」的样本；修正回升机制已在 bedroom 随机视频轮验证（队列 20→10、空 mask 7→0、面积回升）。
- **结论性发现**：SAM2 tiny 在 DAVIS 单目标短片上几乎不真失败 → 复核队列在干净视频上的价值主要是「正确地保持沉默」（5 段共 6 个误报、0 个漏检）；队列真正的用武之地是 **messy 真实拍摄视频**（如 bedroom 手机片）。
- **数据集调研结论**（找"单目标+密集真值+可只下一段+tiny 真会丢"）：MOSE/MOSEv2 按设计就是多目标且 val 只给首帧真值、打包为不可分段流式的 tar.gz/Xet；LVOS V1/V2 多目标、仅 GDrive/Baidu/Kaggle、平均 ~623 帧；SA-FARI 需授权且 RLE 标注。**唯一同时满足全部硬条件的就是已用的 DAVIS-2017 ZIP 里的单目标序列，而它们对 SAM2 太容易**。要拿到真值上的 before/after J&F 回升，只剩两条路：① 人为弱化首帧框制造可控失败（确定可演示，但人造）；② LVOS V1 valid 经 Kaggle 下载后本地筛单目标长视频（真长时漂移，但需 Kaggle 授权 + 大包下载，且长视频务必上 GPU 跑）。

产物：`outputs/davis-blackswan/`、`outputs/davis-drift-straight/`（各含 `evaluation_report.md` 的 Ground-Truth Quality 节）。下一步若要在真值上看「修正回升 J&F」，需要一段 SAM2 tiny 真会丢的更难序列（重遮挡，如 car-roundabout）或人为弱化首帧框。

### 2026-06-12 · MOSE 真值 before/after J&F（首个自然漂移样本）

DAVIS 5 段对 SAM2-tiny 全太容易（见上）。改用 MOSE（crowded/occlusion 硬数据集）：把 26GB `MOSE_release.zip` 下到 **D:盘**，从内层 `train.tar.gz` 流式抽单视频、只跟其中一个实例（GT = `annotation == k`，默认取首帧最大实例）。方法：一趟 tar 流式抽多视频（滤 macOS `._` 垃圾）→ 跑评估 → 用真值框做 GT 派生修正（这批为一次性实验脚本，跑完已清理；方法记录于此，可按需重建）。

视频 **398d41f5**（鸽群，8 个相似目标干扰，60 帧，目标面积 30× 变化）：
- **Baseline**：J&F **0.59**（J 0.61 / F 0.57），逐帧 IoU 多帧跌破 0.5（frame 29 直接 0.0），复核队列报 **30 帧**（area_spike×13、iou_drop×12…）——队列在真漂移上非常活跃。
- **2 个 GT 派生 tight box 修正**（frame 3 漂移起点 + frame 29 最差帧）→ 重传播：**J&F 0.59 → 0.6755（+0.086）**，队列 30→22。worst 帧 IoU 回升：f29 0.0→0.77、f3 0.26→0.77、f18 0.16→0.66。可视化 `D:/MOSE_data/_398d41f5_before_after.jpg`。

**这是第一份「真值上修正回升 J&F」的自然漂移证据**（非人造）：相似目标的鸟群正是 SAM2-tiny 会混淆/跟丢的场景，印证复核队列的价值在 hard 真实素材而非干净基准。MOSE 数据全在 D:盘，未占 C:。

**GPU 扫剩余 4 段（RTX 3060，每段 baseline ~35s）**：

| 视频 | 难点 | baseline J&F | 漂移 | 修正后 |
| --- | --- | ---: | --- | --- |
| 5e9412c0 | 8 目标拥挤 | 0.951 | 无（min IoU 0.80） | — |
| 1fb378d9 | 40× 缩放 | 0.946 | 1 帧(f65) | 0.946→0.956 (+0.011) |
| 0d967fd1 | 3 目标 | 0.913 | 无（min 0.73） | — |
| 6da17988 | 目标消失 51/60 帧 | 0.970 | 无 | 队列报 **54 帧全是误报**（SAM2 正确输出空 mask，几何信号被「出现/消失」过渡触发） |

小结：5 段 MOSE 里只有 **398d41f5（鸽群相似目标）真大漂移并能修正回升**；其余 tiny 都跟得不错。`6da17988` 给出又一条 GT 背书的发现——**目标消失/重现时几何队列会大批误报**（54/60），而 SAM2 的 mask 实际没错。这两条（相似目标会真漂移、消失重现会误报）共同说明：队列要从「几何代理」升级到「SAM2 置信度/语义信号」才能在 hard 场景既抓真漂移又少误报（landing_readiness gap#1）。

**第二批 10 段（GPU，11–14 目标超拥挤）+ bedroom 全量**：又得一个强样本 **23b9a3ea：J&F 0.787→0.924（+0.137，2 个真值框修正）**；fbfb6e30 +0.040；255f86ef/fe470104 修正反而轻微变差（−0.02/−0.01，给部分正确轨迹重播种有时更糟）；d262b7e4 又一例消失场景队列报 54 帧全误报；其余多段干净。**bedroom 全 388 帧上 GPU**（137s，0.35s/帧，CPU 的 ~16×）：队列报 **72/388 帧**，准确圈出 4 段以上跟丢（尾部 ~25 帧目标永久丢失），放过 ~316 稳定帧。15 段 MOSE 平均 baseline J&F≈0.88（方差大），DAVIS 5 段≈0.96。完整汇总已写进 `README.md` 的「实验结果」节。

## 下一步执行清单

1. 准备 10 到 20 段真实短视频，用固定间隔复核和 review queue 复核做对比。
2. 运行 `scripts/generate_eval_manifest_template.py --video-dir data/eval_videos --output data/eval_manifest.template.json --first-frame-dir data/eval_first_frames --overwrite` 生成模板和第一帧。
3. 补齐 `init_box_xyxy`、`object_name`、`expected_challenge`；能人工判断漂移位置的 case 同时补 `expected_review_frames`。
4. 保存为 `data/eval_manifest.json`，运行 `scripts/run_evaluation.py data/eval_manifest.json --calibrate`，比较 sensitive/default/conservative 三档阈值的复核量、命中率和漏检率。
5. 记录真实人工交互次数，逐步替代 `estimated_min_interactions`。
   - **✅ 机制已就位（2026-06-12）**：修正改为**按真实动作计数**——点修正可多点（`x,y;x,y`，runner 应用全部点并带每点 label），`estimated_interactions = 点击数`（框 = 1），`actual_interactions` 汇总真实动作数，取代「每帧算 1 次」。`estimated_min_interactions` 仍作复核前的下界估计；二者在 `comparison.json`/eval 里并列。45 单测全过。
   - **下一步（需真人/UI）**：记录「复核后判定无需修正」（误报 dismiss）这类轻量交互，得到端到端的真实人工成本；目前修正框仍多由 GT 自动生成，非真人点击。
6. 增加 correction 后质量回升验证：轻量指标（mask 面积恢复、队列减少、人工修正帧数）已在首段真实视频上验证；IoU/J&F 评分已实现，下一步是给评估 case 准备真值 mask 目录（`ground_truth_mask_dir`）跑出 before/after J&F。
7. **（数据已给出明确方向，建议作为下一步 #1）队列从几何启发式升级到 SAM2 置信度/语义信号**。本批 5+15 段真值实验证明：纯几何队列在干净视频几乎不漏检但会误报，且在「目标消失/重现」上大批误报（`6da17988`、`d262b7e4` 各 54 帧全误报），同时在相似目标拥挤场景能抓真漂移（鸟群 398d41f5、14 目标 23b9a3ea，修正可回升 J&F +0.09~+0.14）。做法：`sam2_video_predictor.propagate_in_video` 本就返回每帧 `mask_logits`，把它的置信度（如前景 logit 分布/峰值）暴露到 `sam2_runner` → `pipeline`，用「低置信 + 几何信号」联合判据替代纯几何——目标是消失重现少误报、相似目标早预警。这是当前性价比最高的一步。
   - **✅ v1 已完成（2026-06-12）**：`Sam2VideoSegmenter.last_confidences`（峰值 object logit 的 sigmoid）→ `pipeline` 写入 `mask_confidences`、新增 `low_mask_confidence` 信号、每个队列项带 `confidence` 供分诊；新增 `ReviewPolicy.low_confidence_threshold`；UI 加 `CONF-` 样式；向后兼容（无置信度的 segmenter 退回纯几何）；**41 单测全过**。
   - **GPU 实测验证（两类失败互补）**：`6da17988`（目标消失）present 帧 conf=**1.00** / absent 帧 conf=**0.02**，50/50 低置信帧确为真消失 → 置信度**精准识别消失/丢失**；`398d41f5`（相似目标，SAM2 自信跟错鸟）drift 帧 conf 仍 **0.985**、零低置信帧 → 置信度**抓不到「自信跟错」**，要靠几何信号。**结论：两信号互补**——置信度管丢失/消失（并标出可抑制的真消失误报），几何管自信跟错。
   - **✅ v2 已完成（2026-06-12）**：`ReviewPolicy.collapse_low_confidence_runs`（默认开）把连续「空 mask + 低置信」帧合并为**一次 onset 交互**、标 `status: low_confidence_empty`，帧仍保留在队列里可见。GPU 实测 `6da17988`：队列 54 帧不变，但**估计交互 54 → 5**（50 帧消失只算 1 次），interactions/min 同步下降。**42 单测全过**。
   - **✅ v3 已完成（2026-06-12）**：统一 per-frame `review_score`（0–1，`pipeline.REVIEW_REASON_WEIGHTS` 为单一真相源）合并几何原因权重 + 置信度/合并状态，挂到每个队列项；UI 改读 `review_score` 不再自算。真机验证：`398d41f5` 真漂移帧 f29（IoU 0.0）排第 1，`6da17988` 49 个合并消失帧排 0.15 垫底。43 单测全过。
   - **诚实边界（不做）**：用 SAM2 occlusion score 区分「消失」vs「在场但跟丢」**做不到**——两种情况 SAM2 都是低置信/抑制 mask（它若知道目标在就不会丢）。要可靠区分得接外部检测器在空帧上重定位，超出当前范围。`review_score` 已是可调权重（改 `REVIEW_REASON_WEIGHTS` 即可），需要再做成 `ReviewPolicy` 字段时再说。
8. **GPU 验证（P1 收尾）— ✅ 已完成（2026-06-12）**：本机其实有 **NVIDIA RTX 3060 12GB**（之前一直是 CPU-only torch 才没用上）。装 `torch==2.5.1+cu124 torchvision==0.20.1+cu124`（`--index-url .../cu124`，注意要带 `+cu124` 本地版本号或 `--force-reinstall`，否则 pip 会认为 `2.5.1+cpu` 已满足 `==2.5.1` 而跳过）后 `cuda_avail=True`。实测 MOSE 60 帧视频 baseline **~35s/段（CPU 约 7 分钟，≈12× 提速）**，`bfloat16` autocast 正常，显存占用远小于 12GB（`offload_video_to_cpu=True`）。一次性实验脚本经 `MR_DEVICE=cuda` 切换设备（已随清理删除）。生产代码本就 `--device cuda` 默认，无需单独 GPU 版本。后续把更难/更长序列（bedroom 全 388 帧、更多 MOSE）放 GPU 跑成本已可接受。

## 不做清单

- 不做完整 CVAT 替代品。
- 不做文本提示和自然语言检索，除非复核闭环已经跑通。
- 不做视频编辑特效。
- 不把重点放在 SAM2 复现本身，重点是传播后的自检和 human-in-the-loop 复核。
