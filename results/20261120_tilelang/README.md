# TileLang 试点 · 阶段 0+1 判决（GEMM 校准对决赛）

2026-11-20 B 机 · feat/tilelang-pilot · 依据 docs/交付-B机-TileLang试点单.md

## 判决：X = 116T ≥ 100T → **试点成功，阶段 2 立项**

| 口径（与 triton 56T 锚完全同法） | 峰值 |
|---|---|
| TileLang（tilelang-metax 源码构建，USE_MACA=ON） | **116T** |
| triton 3.0+metax `tl.dot`（d7 锚，256,64,64,w8,s1） | 56T |
| 倍率 | **2.07×** |

单位说明：两行同为 d7 口径的 MAC 计数（`104 CTA×1024 iter×BM·BN·BK / t`），
apples-to-apples；折 FLOPS 翻倍（232 vs 112）。**沐曦 triton 3.0 的 tl.dot
天花板被 TileLang codegen 绕过**——「计算密集 kernel 自研」门重开。

## 校准表（9 配置，全 graph 口径）

| BM,BN,BK,warps | µs | MAC-T |
|---|---|---|
| 128,32,128,w8 | 501.1 | 111 |
| 128,64,128,w8 | 973.1 | 115 |
| 64,64,128,w4 | 972.1 | 57 |
| 128,128,64,w8 | 984.2 | 113 |
| 256,64,64,w8 | 976.8 | 114 |
| 128,256,64,w8 | 1928.3 | **116** |
| 256,128,64,w8 | 1926.6 | **116** |
| 64,128,128,w8 | 982.7 | 114 |
| 128,64,64,w4 | 972.1 | 57 |

规律：① w8 全档 111-116T、w4 全档 57T——tilelang 的 gemm 在本后端降出
8-warp MMA 形态，warps 才是第一杠杆（triton tl.dot 的天花板恰好停留在
tilelang w4 的水平，推测 triton 没降出该形态）；② 各档时间随 BM·BN·BK
精确线性（自洽校验过：同 MAC 数不同形状时间一致）；③ 峰值对形状不敏感
（111-116T 平台）→ 这是 codegen/硬件平台，非某形状特异。

## 阶段 0 记录（环境，用时 ~36min，盒内 ≤2h）

- pip 版 tilelang 0.1.15 为 CUDA 轮子，不驱动 MACA → 走试点单预案：
  **tile-ai/tilelang-metax**（TileAI×沐曦官方合作适配，README 明言
  Tested Devices 含 C500；ghfast 镜像 clone）源码构建 `USE_MACA=ON cmake`
  + `make -j64`（128 核 ~6min，坑：缺 cython 补装；z3-solver 单独 --no-deps
  装 4.15.4.0 防 numpy 2.5.5 连带升级——厂商栈 numpy 锁 1.26.4）。
- 判据达成：import 与 torch 共存 ✓，quickstart matmul 数值对 ✓（assert_close
  无异常 + latency 0.109ms），版本串 `0.1.15+cuda.git90d49f35`。
- 运行形态：`PYTHONPATH=/tmp/tilelang-metax`（dev root 加载 build/lib）+
  `LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib`——**不 pip 安装**，
  与 vllm/flaggems 栈零接触（本机 vllm_fl 为 site-packages 拷贝，见 D13
  WIRING.md 踩坑 3，避免再引入包管理漂移）。源码树保留在 /tmp/tilelang-metax
  （含子模块，~1.5G；阶段 2 若立项建议移入持久位置）。

## 方法论对齐声明（铁律）

- 复刻 scripts/d7/gemm_calib.py（commit a18e64a，feat/d7-prefill-attn）：
  单 kernel 单 grid (104,)、iters=1024 数据驻留循环 gemm、TFLOPS 同式。
- 计时 bp.graph_time_us（CUDA graph 3×5 replay 中位）；9/9 配置均 graph
  口径捕获成功（无需事件口径回退）。
- 工作集驻留 shared/寄存器 = L2-warm 等效口径，与 d7 相同——绝对值只用于
  同口径对决赛，不外推 serve（铁律 3）。

## PROVENANCE

- 机器：沐曦 C500 64GB（B 机），GPU 独占（mx-smi 空闲，无并发测量）
- MACA 3.7.0.38 / torch 2.8.0+metax3.7.0.7 / tilelang-metax dev@90d49f35
- 原始输出：calib_run1_raw.log；脚本：scripts/tilelang/gemm_calib_tilelang.py
- 判决表回执：X≥100T → 阶段 2（原型 kernel 三选一：① norm+quant TileLang 版
  ② decode split-KV attention ③ prefill attention——116T>96T ③技术上也开放，
  选题待用户/A 机裁定）
