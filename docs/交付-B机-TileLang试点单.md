# 交付文档 · TileLang 试点单（B 机）

> 目的：验证 TileLang 能否绕过沐曦 triton 3.0 的 `tl.dot` 编译器天花板（本征 ~56 TFLOPS，你已用
> GEMM 校准钉死，台账 §9）。这是 C500 上重开"计算密集 kernel 自研"门的唯一自主通道
> （MI300X 先例：TileLang 手写 kernel 比 Triton-AMD 后端快 ~1.53×；沐曦自家揭榜挂帅赛主推
> TileLang+MXMACA，生态支持预期最好）。**时间盒：试点 1 天出判决，不无限投入。**

## 阶段 0 · 环境拉起（≤2h）

```bash
pip install tilelang  # 或 git+https://ghfast.top/https://github.com/tile-ai/tilelang
python3 -c "import tilelang; print(tilelang.__version__)"
```
- 若 pip 版不支持 MACA：查 tilelang repo 的 MACA/沐曦 后端分支（沐曦赛题官方用的是 MXMACA 适配版，
  `github.com/MetaX-MACA` 组织下可能有镜像/补丁；ghfast 镜像拉取）；
- 需要 torch 2.8+metax 共存（tilelang 通常以 torch 扩展形态工作，验证 `import tilelang, torch` 不冲突）；
- 烟测：vector add + 简单 matmul 跑通、数值对。

## 阶段 1 · GEMM 校准对决赛（核心判决，≤3h）

复刻你自己的 `scripts/d7/gemm_calib.py` 方法论（104 CTA×1024 iter、graph/事件口径、L2 说明标注），
同形状集（BM/BN/BK × warps × stages 网格）跑 TileLang matmul：
- **判决表**：triton 峰值 56T vs TileLang 峰值 X T（同卡同口径）
- 判据：**X ≥ 100T**（mctlass w8a8 水位的一半以上）→ 试点成功，阶段 2 立项；
  X < 70T → 判死归档（工具链同样不行，台账记录关门）；
  70-100T → 灰区，向用户报告后再定

## 阶段 2 ·（若过线）原型 kernel 三选一（按性价比，≤1 天）

1. **norm+quant 融合的 TileLang 版**（对标 D13 triton 版 1.10-1.16×，验证 elementwise 类是否也有优势）；
2. **decode split-KV attention TileLang 版**（对标 MACA flash 的 83-88% 带宽利用——只求 ≥60% 即可
   与 int8-KV 组合打出净赢，数学：int8 字节减半 × 0.6 带宽 = 1.2× vs bf16 flash）；
3. **prefill attention**（96T 平台破格——若 TileLang GEMM 校准 >96T 才值得选这项）。

## 纪律与交付

- 数值门照旧（位精确/分布等价 T0 + 105 题终判）；产物落 `results/<日期>_tilelang/`（校准表+判决 README）；
  分支 `feat/tilelang-pilot`；台账 §9 追加行；结论无论生死都回任务板。
- 红线不变：serve 冻结、bench 原件、官方测需用户批准。
- 提示：沐曦 op_optimization 赛题 repo（github.com/MetaX-MACA/op_optimization）内有 TileLang 官方样例与
  C500 调参先例，可作参考（勿抄赛题答案，只借工具链用法）。
