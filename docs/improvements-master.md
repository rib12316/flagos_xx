# flagos_xx 改进总账（D0→D4 全量整合，2026-11-19）

单文件汇总所有已整合进生产（vllm-plugin-FL）的改进、待门禁项与结案项。
数据可回查各阶段 results/ 与 docs/；本文件是唯一入口。

## 0. 端到端成绩线

| 配置 | 4K 场景 (tok/s) | 16K 场景 (tok/s) | 口径 |
|---|---|---|---|
| 纯净起点（无 D0/D1/D2） | 1879.68 | 862.06 | 官方 run |
| + D0 attention launch 调优 | 2658.72（+41.5%） | 1262.66（+46.5%） | 官方 run |
| + D1 watermark=0.06 | 2668.42（+0.4%） | 1276.62（+1.1%） | 官方 run |
| + D2 W8A8 量化线 | 2838.46 / 2835.73（+6.4%） | 1345.90 / 1346.04（+5.4%） | 官方 run1/run2 |
| + D3②（默认态） | 2741.07（−3.4%*） | **1481（+10.0%）** | D3 官方（11-16/17 战役，results/d3_official_1116_2222/） |
|  D3 臂 B（②+⑤+①TD，终选） |  **4K 3414.58（+20.3%）** |  **16K 1570.62（+16.7%）** | 同上；**最优候选，105 题已过（0.9714）** |
| D3 臂 A（②+⑤+④int8） | 2.5 轮不足 | 1212.4（**−9.9%，判负**） | 同上；int8 出局 |
| 全项目 vs 起点（天数） | **+81.7%（B 臂）** | **+82.2%（B 臂）** | |
| 沐曦纯净基线（bf16，11-19 实测） | 3786.30 | 5583.49 | 官方 4 轮丢首；L3=98.1%（10-30 bf16 基线）；`results/20261119_d4_metax_port/baseline/` |
| 沐曦终局（+ D4 mctlassEx w8a8 承载） | **6399.21（+69.0%）** | **7509.87（+34.5%）** | 官方形态零附加 env、无 compilation-config；105 题 **98.1%**（`results/20261119_d4_metax_port/final/SUMMARY.md`） |

*default 4K −3.4% 未归因（轮 3-4 强于 D2，轮 1-2 暖机偏慢=夜间方差嫌疑）。
EngineCore 死亡/悬挂贯穿 D3 战役（9 serve 6 次）——官方提交须每场景独立
新鲜 serve。详 `results/d3_official_1116_2222/README.md`。

## 1. 已整合·默认生效（无需任何 env）

| 改进 | 机制 | 文件 | 实测收益 |
|---|---|---|---|
| D0 attention launch 表 | per-shape（路径×bs档×kv档）注入 num_warps/stages：decode w1s1、prefill w2s1 | `attn_launch_config.py` | decode 3D 1.78-1.85×、prefill 1.50×（bit-exact） |
| D2 W8A8 量化 ckpt | SmoothQuant α=0.5 权重 int8 per-channel + 激活 int8 per-token | `/workspace/MiniCPM5-2B-w8a8-v2` | 官方 +6.4%/+5.4%（vs main）；线性层贴 vendor 顶 |
| D2 P0 融合 decode GEMM | `_w8a8_fused_gemm_kernel` 小 BM tile（M=64/128 档） | `w8a8/int_mm.py` | 逐层 1.16-1.62×，ITL −19.4%（M=64） |
| D2 P1 lm_head int8 | lm_head 转 int8（ckpt 内） | ckpt + 转换开关 | −267MB 显存，parity 0.9667 |
| D2 P4 epilogue 融合 | `_run_fused_epilogue`（dequant+scale 融合） | `w8a8/int_mm.py` | 随 P0 计入 |
| D3 ② 采样过滤 grid | `_topk_topp_kernel` grid 16→64 programs（行间独立=逐位等价） | `sample_filter_mode.py` | 过滤 3.7→1.1ms/步，decode ITL **−9~12%**，对拍 16/16 |
| D1 调度默认值 | mnbt=8192 / watermark=0.06 / APC on（config 级采纳） | platform.py | D1 判决采纳值 |
| D4 沐曦 w8a8 metax 承载 | mctlassEx `w8a8_scaled_mm_azp`（vendor='metax' 路由 + [N,K] 零拷贝布局 + custom_op 黑盒化 fullgraph 安全；int_mm 融合档沐曦清空=smem 超限） | `vllm_fl/quantization/w8a8/mctlass_mm.py` | 沐曦终局 **+69.0%/+34.5%**（6399.21/7509.87 vs 沐曦基线） |

## 2. 已整合·env 门控（**①⑤ 已于交付封板 `4439398` 翻转为默认开**，`=0` 为显式回退；④ 保持默认关）

| 改进 | 机制 | env 开关 | 实测收益 | 对拍 |
|---|---|---|---|---|
| D3 ① TD 加载路径 | Q/K/V tensor descriptor 2D 块读（强制 use_td=True） | `VLLM_FL_ATTN_TD=0`（回退门，**交付默认开**） | decode ITL −6.4%（探针 attn 1.20-1.55×，官方 kv 区更高） | **0/16** |
| D3 ⑤ prefill TILE=16 | kv 循环块 32→16（本卡「多小 CTA」规律） | `VLLM_FL_ATTN_TILE=0`（回退门，**交付默认开**） | prefill 代理 **+12.8%**（探针 1.10-1.11×） | **0/16** |
| D3 ④ KV int8 扩池 | KV cache bf16→int8 per-token-head，池 ×1.94（588K→1.14M） | `VLLM_FL_KV_INT8=1`（默认关） | 16K 同驻 33→64、0 抢占；decode 税 +4%；prefill 税未观察到 | **1/16** |

**组合约束：①×④ 禁止同开**（实测组合 decode 比默认差 16%，TD×descale 互毁）。
推荐门禁臂：A=②+⑤+④（16K 主打）或 B=②+⑤+①（4K/确定性主打），二选一。

## 3. 结案/死案（有数据档案，勿重开）

| 项 | 死因 | 档案 |
|---|---|---|
| D3 ③ 段数自适应 | 收益区 kv8K 官方到不了；bs32/kv16K 倒挂 0.92×；捕获图无法按 kv 分档 | `results/d3_dev/item3_segments/` |
| D3 ⑥ 自写 split-KV kernel | 机理假设被零成本探针证伪（decode TILE=16 已最优、重分段无益）；残留 1.7× 与 TD 重叠，终局门后重估 | `results/d3_dev/item6_splitkv/dtile_sweep.csv` |
| D3 prefill kv-split（#56148 式） | **未探区**（非死案）：TILE 只收 1.11×<阈值，kv-split 是 4K 场景下一个真算法项，2-3 天级 | 计划 ⑤-3 设计备忘 |
| D2 P2 KV int8 提速 | kernel 层净负（attention 非带宽 bound）→ 仅容量角度在 D3④ 复活 | `results/20261114_d2_kv_probe/` |
| D2 P3 W4A8 | 位精确 0.24-0.26×，int4 原语两层全无 | 同上 |
| D4 ⑥ split-KV 接线（iluvatar 侧） | 判负：接线删除、env 彻底失效防误开（`27c8ab5`）；kernel 文件与单测保留为研究档案；**沐曦侧 hop 独立不受影响** | `results/20261119_d4_metax_port/splitkv_iluvatar_prompt.md` |
| 投机解码 / LCF 抢占 / block_size / max_num_seqs | 高并发负收益 / 0 抢占无对象 / <1% / 从未触顶 | `docs/bottleneck-research-report.md` |

## 4. 回退矩阵（全 env，默认态=基线+②）

```
VLLM_SAMPLE_FILTER=0          关 ②（默认 1）
VLLM_FL_ATTN_LAUNCH=0         关 D0 launch 表（默认 1）
VLLM_FL_INTMM_FUSED_DECODE=0  关 P0 decode 融合（默认 1）
VLLM_FL_INTMM_FUSED_GEMM=0    关 P0 prefill 融合（默认 1）
VLLM_FL_ATTN_TD=0             关 ①（交付封板后默认开，=0 显式回退）
VLLM_FL_ATTN_TILE=0           关 ⑤（交付封板后默认开，=0 显式回退）
VLLM_FL_KV_INT8=1             开 ④（默认关）
VLLM_FL_SAMPLE_FILTER_PROGRAMS=64  ② 的 grid 上限
VLLM_FL_ATTN_LAUNCH_TABLE=<json>   D0 表热换（A/B 用）
```

## 5. 纪律与门禁

- 官方双场景 bench + 105 题精度：**仅用户明示后开测**（六项已完成，条件满足）。
- 每次改插件后 `pip install --no-deps --no-build-isolation ./vllm-plugin-FL` 再 serve。
- EngineCore 有间歇静默死亡前科（④ 一次未复现）——官方 bench 用新鲜 serve、
  每场景一窗，勿长跑串联。
- 对拍方法：`scripts/d3/parity_check.py`（16 条 greedy，基线 `results/d3_dev/
  parity_baseline.json`，跨 serve 可复现已证）。

## 6. 阶段文档索引

D0-D1: `docs/track-a/b/c-*.md` · D2: `docs/d2-w8a8-delivery.md`（量化线）+
`d2-quant-blocking-analysis.md`（死案）· D3 深挖: `results/20261115_bottleneck_deepdive/`
+ `docs/bottleneck-research-report.md`（调研）· D3 开发: `docs/d3-dev-delivery.md` +
`results/d3_dev/DEVLOG.md`（逐项过程）· 快诊: `docs/bottleneck-quickdiag-playbook.md`。
D4 沐曦移植: `results/20261119_d4_metax_port/`（final/baseline/PROVENANCE）·
双平台终局与激活方式: 根目录 `DELIVERY.md`（分支 `delivery/s2-final` = 唯一事实源）。
