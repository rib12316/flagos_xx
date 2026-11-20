# C500 四项融合整包 quick

2026-11-20，A 机，MiniCPM5-2B-w8a8-v2。基线先跑，整包后跑，各用一个新 serve。整包代码 `6c5eaa2`，入口 `scripts/fusion_bundle/quick.sh`，显式设 `VLLM_FL_FUSION_BUNDLE=1` 和 `VLLM_FL_D13_M_FUSE_MAX=2048`。M=2048 是编译预热形状。先前的 512 配置虽配对 83 处，实际预热生成图仍调用分开的 norm 和 quant。

| 请求数 | 全关后三轮 Total tok/s | 全开后三轮 Total tok/s | 中位数变化 |
|---:|---|---|---:|
| 32 | 5892.74 / 5880.20 / 5919.57 | 6013.66 / 6061.27 / 6074.96 | +2.86% |
| 64 | 7016.42 / 7032.84 / 7013.89 | 7106.88 / 7078.07 / 7107.57 | +1.29% |

冻结参数，并发上限 64，固定 4096 输入、1024 输出，每臂每形状 4 轮，首轮预热。全部 16 轮足额成功。`baseline.csv` 和 `bundle.csv` 保留逐轮吞吐与延时。一次顺序对照的结果支持作为候选，跨启动重复性待验证。

服务记录中，基线 D13 disabled、fused_add_rms_norm=native。全开有 `paired 83 sites (m_fuse_max=2048, buf_rows=2048)`、优先级 `['vllm_c', 'native']`、`F-c grouped decode out active`，编译产物含 `torch.ops.vllm.fl_rope_qk_cat.default` 及 `torch.ops.vllm_fl.d13_norm_quant.default`。A-H4 的 priority 描述可选路径，不能据此计数其动态调用次数。

`baseline-smoke.json`、`bundle-smoke.json` 保留相同输入的贪心响应，每例 24 个生成 token。英文续写、算术、中文、代码的首个分叉位置分别是 21、2、11、全同（从 1 起）。第一臂未显式传请求 seed，整包臂传 0，两臂均 temperature=0。官方精度尚待评估，独立重启的同臂对照也有助于定位分叉。整包保持默认关。
