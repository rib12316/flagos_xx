# F-c（copy 消除）产物档案 · 2026-11-20

分支 feat/fc-copy-attribution（agent-fc，1h 盒两轮延长至 13:5x）。

## 判决：**主站已归因（可删型确认）；直写补丁负判决封存（休眠逃生门）**

- 归因（成立）：`_copy_kernel_kernel_rank_3` 稳态 decode **42.0 次/步 = 每层 1 次**，
  [32,2048] bf16 × 3.30µs = **0.14ms/步**，100% 为 attention 输出→o_proj 输入的
  staging 拷贝（前驱 flash_fwd_splitkv_combine、后继 o_proj quant；GPU 时间线
  邻接法，`scripts/fc/trace_copy_attribution.py` 可重复）。`linear.py:117`
  的 `.contiguous()` 反证排除。
- 补丁（负判决）：MACA `flash_attn_2_cuda.fwd_kvcache` 第 13 参（python wrapper
  恒传 None）为 out 参，C++ 检查链存在（dtype/is_cuda/stride(-1)==1/sizes=={q0..q3}），
  但**该构建（md5 0953e2b9d24f3276）的 out 形状合同运行期不可满足**——
  探测矩阵 V1（out 逐维==q.sizes()）仍被拒，V7/V8 证明 out 通道与检查链真实可达。
  vendor 更新 .so 后翻 `VLLM_FL_FLASH_KVCACHE_DIRECT_OUT=1` 即可复活重测。

## 文件

| 文件 | 内容 |
|---|---|
| probe_matrix.txt | 探测矩阵九变体原始读数（判死证据） |
| t0_direct_out.log | T0 v1 失败日志（3D out 误诊轮） |
| t0_v2.log | T0 v2 失败日志（4D q 形仍拒，触发矩阵探测） |

判死证据链完整版见 `scripts/fc/ATTRIBUTION.md` 终章。
