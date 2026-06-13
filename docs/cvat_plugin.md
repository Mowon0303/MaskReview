# MaskReview → CVAT 插件设计 (P8)

> 状态：设计文档（代码未实现）。目标环境：**自托管 CVAT（Docker compose）**。
> 配套：[research_plan.md](research_plan.md) Phase 3 / M3、[landing_readiness.md](landing_readiness.md)、[PLAN_TRACKER.md](../PLAN_TRACKER.md) P8。

## 1. 目标与定位

MaskReview 是**自动预标注 + 难帧分流引擎**；CVAT 是**成熟的人工修正 UI**。插件不替代 CVAT，也不把 MaskReview 变成通用标注平台——它只把"复核闭环"接进 CVAT 的真实标注工作流：

> 标注员在 CVAT 画首帧框 → 插件读取 → 跑 SAM2 双向传播 → 把 mask 作为预标注写回 task、把**低置信/异常帧**作为 **CVAT Issues** 标出来 → 标注员只处理这些 issue 帧 → 导出修正后的 annotation + 真实交互次数。

**不做**：完整 CVAT 替代品、CVAT 前端二次开发、Nuclio serverless 推理函数（那是另一条路，见 §10）。本插件走 **cvat-sdk（REST）** 这条更轻的集成路径。

## 2. 概念映射

| MaskReview | → | CVAT |
|---|---|---|
| 从 task 拉取的视频帧 | ← | task 的 data（**帧来源以 CVAT 为准**，见 §6.1） |
| 首帧 `init_box_xyxy` | ← | task 第一帧上标注员画的 rectangle shape |
| SAM2 per-frame masks | → | task 预标注（mask shape / COCO RLE 导入） |
| `review_queue` 每个难帧（reason + `review_score` + recommended_correction） | → | 该帧上的一个 **CVAT Issue**（带评论） |
| `estimated_min_interactions` | → | 与"已解决 issue 数 / 导出 annotation 的实际改动"对比 |
| `mask_oversized` / `low_mask_confidence` / `likely_occlusion` 等状态 | → | issue 评论里的标签，决定 issue 的优先级措辞 |

## 3. 架构

新增独立目录 `src/cvat/`（本地 MVP 稳定后再开，见 PLAN_TRACKER P8）：

```
src/cvat/
  export.py     # 纯离线：mask→RLE/COCO；review_queue→issues 负载。可单测，零 CVAT 依赖。
  client.py     # 薄 cvat-sdk 封装：拉 task / 推 annotation / 建 issue / 拉回修正。需要活的 CVAT。
  sync.py       # 编排：把 export + client + 现有 pipeline 串成完整闭环。
scripts/cvat_sync.py   # CLI 入口：--task-id N --host ... --pull/--push/--export
```

**离线 / 在线 分层是关键**：`export.py` 不 import cvat-sdk，只做格式转换 + 数据建模，能在没有 CVAT 的情况下完整开发并单测（这是 landing_readiness 要求的前置层）。`client.py` 是唯一碰网络/SDK 的地方，尽量薄。

## 4. 环境（自托管 Docker CVAT）

```bash
# 1. 起 CVAT（一次性）
git clone https://github.com/cvat-ai/cvat
cd cvat && docker compose up -d        # 默认 http://localhost:8080
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

# 2. 装 SDK（进 maskreview 的 .venv）
pip install "cvat-sdk"                  # 锁定版本，API 跨版本会漂（见 §10）

# 3. 配置（用环境变量，别硬编码）
export CVAT_HOST=http://localhost:8080
export CVAT_USER=...   CVAT_PASSWORD=...   # 或 token
```

CLI/客户端从这些环境变量读 host + credentials。

## 5. 数据流（完整闭环）

```
[标注员] 在 CVAT 建 task（上传视频）+ 第一帧画 rectangle
        │
        ▼
cvat_sync --task-id N --pull
  client.pull_task(N):  下载帧（§6.1） + 读首帧 rectangle → init_box_xyxy
        │
        ▼
  pipeline.run(VideoSegmentationRequest(frames=task_frames, init_box_xyxy, init_frame_index))
  → masks / review_queue / metrics            （复用现有引擎，零改动）
        │
        ▼
cvat_sync --task-id N --push
  export.masks_to_coco(masks)        → annotations.json/zip
  client.import_annotations(N, ...)  → task 预标注（异步，poll requests）
  export.queue_to_issues(review_queue) → issues 负载
  client.create_issues(N, issues)    → 每个难帧一个 issue
        │
        ▼
[标注员] 在 CVAT 里按 issue 列表只修这些帧，逐个 resolve
        │
        ▼
cvat_sync --task-id N --export
  client.export_dataset(N, "COCO 1.0") → 修正后 annotation
  export.interaction_report(before=review_queue, after=resolved_issues/exported)
  → 真实交互次数 vs 估计，写 comparison
```

## 6. 格式与对接细节

### 6.1 帧对齐（最关键的正确性点）
CVAT 把视频解码成自己索引的帧序列；MaskReview 也会抽帧。两者**必须用同一套帧**，否则 mask/issue 的 frame number 对不上。
**方案**：帧来源以 CVAT 为准——`client.pull_task` 通过 `task.get_frame(i)` / 下载 chunk 把 task 的帧落到本地，MaskReview 在**这些帧**上跑。这样 mask dict 的 key 和 issue 的 frame 天然对齐 task 的帧号。
（不要各抽各的：fps/抽帧策略不同会错位。）

### 6.2 读首帧框
`client.pull_task` 拉 task annotations，找第 0 帧（或 `init_frame_index` 帧）上的 rectangle shape，转成 `init_box_xyxy=(x1,y1,x2,y2)`。若该帧没有 rectangle → 报错提示标注员先画一个，或回退到 CLI 传入的框。

### 6.3 masks → CVAT 标注（两条路，先简后精）
- **路 A（推荐先做）：COCO 实例分割导入。** 每帧二值 mask → bbox + RLE（`pycocotools.mask.encode`），组装成 COCO instance segmentation JSON，`task.import_annotations("COCO 1.0", file)` 导入。鲁棒、格式标准、CVAT 原生支持。
- **路 B（精修）：CVAT 原生 mask shape。** 低层 `update_annotations` 写 `LabeledShape(type="mask", points=<CVAT RLE>, frame=i)`。CVAT RLE = `[run-lengths..., left, top, right, bottom]`（bbox 内行优先、从背景起算）。能在 CVAT UI 里当 mask 直接编辑，但要手写 CVAT 专有 RLE 编码。
- MaskReview 已经产出 per-frame PNG，也可走 `"Segmentation mask 1.1"` 导入，但那是**语义类**掩码（非实例），单目标够用、多目标不行。
- **导入是异步的**：`POST /api/tasks/{id}/annotations` 起后台任务，要 poll `/api/requests/{rq_id}` 等完成。封装在 `client.import_annotations` 里。

### 6.4 review_queue → CVAT Issues
CVAT 的 **Issue** 挂在 job + frame + position（帧上一个矩形）+ comments 上，正好就是"写回 review queue"。
每个 `review_queue` 项映射成一个 issue：
- `job`：task 的 job id（视频 task 通常单 job，`task.get_jobs()[0]`）。
- `frame`：`item["frame_index"]`。
- `position`：用 mask 的 bbox（或 diagnostics 里的框）当 issue 矩形。
- comment：`f"[{review_score:.2f}] {reasons} — 建议: {recommended_correction}"`，例如 `"[0.92] mask_oversized — 建议: tight box 重新框定"`。
- 按 `review_score` 降序建，让真问题排前面；被 damping 压低的 `likely_occlusion` 帧用更弱的措辞甚至跳过（可配阈值）。

Issues 走**低层 API**：`client.api_client.issues_api.create(...)` + `comments_api`（高层 SDK 对 issue 包装不全，见 §10）。

### 6.5 导出 + 交互统计
- `task.export_dataset("COCO 1.0", file)` 拉回标注员修正后的 annotation。
- 交互统计：对比导出结果 vs SAM2 原始 mask（改动的帧数），或直接数已解决的 issue 数；和 `estimated_min_interactions` 对比，复用现有 comparison.json 思路。

## 7. 接口草图

```python
# src/cvat/export.py  —— 纯离线，无 cvat-sdk
def masks_to_coco(masks: dict[int, np.ndarray], label: str) -> dict: ...
def mask_to_cvat_rle(mask: np.ndarray) -> tuple[list[int], tuple[int,int,int,int]]: ...  # 路 B
def queue_to_issues(review_queue: list[dict], *, min_score: float = 0.0) -> list[IssuePayload]: ...
def interaction_report(before: list[dict], after_resolved: int) -> dict: ...

# src/cvat/client.py  —— 唯一碰 SDK/网络的地方
class CvatBridge:
    def __init__(self, host: str, credentials: tuple[str, str]): ...   # make_client
    def pull_task(self, task_id: int) -> PulledTask: ...               # frames + init_box_xyxy
    def import_masks(self, task_id: int, coco_path: Path) -> None: ... # import_annotations + poll
    def create_issues(self, task_id: int, issues: list[IssuePayload]) -> None: ...
    def export_annotations(self, task_id: int, out: Path) -> None: ... # export_dataset

# src/cvat/sync.py  —— 编排，串起 pipeline + export + client
def sync_pull(...) / sync_push(...) / sync_export(...)
```

参考用法（高层 SDK，签名以你装的版本为准）：
```python
from cvat_sdk import make_client, models
from cvat_sdk.core.proxies.tasks import ResourceType
with make_client(CVAT_HOST, credentials=(CVAT_USER, CVAT_PASSWORD)) as client:
    task = client.tasks.retrieve(task_id)
    task.import_annotations("COCO 1.0", "annotations.zip")   # 异步，内部 poll
    task.export_dataset("COCO 1.0", "out.zip", include_images=False)
```

## 8. 实施阶段（offline-first）

| 步 | 内容 | 依赖 CVAT? | 验证 |
|---|---|---|---|
| 1 | `export.py`：masks→COCO、queue→issues 负载、interaction_report | 否 | 单测（mask→RLE 往返、issue 映射、边界） |
| 2 | `client.py`：pull/import/issues/export，薄封装 + 异步 poll | 是（本地 Docker） | 对本地 CVAT 跑一个真 task |
| 3 | `sync.py` + `scripts/cvat_sync.py`：完整闭环 CLI | 是 | 端到端：建 task→pull→run→push→人工修→export |
| 4 | 交互统计接入 comparison / 评估报告 | 是 | 数字对得上 estimated |

**先做步 1**（你选的"先出设计文档"之后的自然下一步）：纯离线、可单测、是后续所有的前置。

## 9. 测试策略
- **离线（步 1）**：单测 `export.py`——mask→COCO RLE→解码 IoU≈1；`queue_to_issues` 的 frame/position/comment 正确；`min_score` 过滤；空队列。零网络。
- **在线（步 2+）**：对本地 Docker CVAT 跑集成测试（建临时 task→push→断言 task 上有 N 个 mask + M 个 issue→export 往返）。标 `@skipUnless(CVAT_HOST 可达)`，CI 默认跳过。

## 10. 风险 / 待确认（实现时核对）
- **SDK 版本漂移**：`create_from_data` / `import_annotations` / issues 低层 API 的签名跨 cvat-sdk 版本会变。锁版本，实现前对着 `cvat_sdk` 实际签名核对一遍。
- **已知坑**：高层 `create_annotations(location='local')` 在部分版本有 FileNotFoundError（cvat-ai/cvat#9576）；COCO 导入有过 "bytes not JSON serializable"（#8215）。用文档推荐的 `import_annotations` + 多部分上传，遇到再绕。
- **Issue 高层封装不全**：issue/comment 多半要走 `api_client.issues_api` / `comments_api` 低层。
- **mask shape vs dataset 导入**：路 A（COCO 实例）鲁棒但 mask 进 CVAT 后是导入态；路 B（原生 RLE shape）可直接编辑但要手写 CVAT RLE。先 A 后 B。
- **帧对齐**（§6.1）：务必以 CVAT 帧为准，别各抽各的。
- **双向种子帧**：若用 `init_frame_index>0`，issue/mask 的帧号仍是 task 全局帧号，无需特殊处理；但要注意 seed 帧之前若目标不存在，反向 mask 可能是虚的（会被 `mask_oversized` 标成 issue，符合预期）。

## 11. 备选路径（不在本设计内）
CVAT 还能把 SAM2 部署成 **Nuclio serverless 交互函数**（点选式自动分割）。那是"SAM2-in-CVAT 交互标注"，**不是** MaskReview 的"传播后复核分流"价值，且要运维 Nuclio。本插件不走这条；如果将来要交互式点选，再单独评估。
