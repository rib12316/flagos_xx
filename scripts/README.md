# scripts/ · 一键脚本规划

按路线图 P0′/P1′ 落地，全部脚本要求：stdout 单行 RESULT 可 grep、退出码非零即失败、产物落 `../results/`。

| 脚本 | 用途 | 状态 |
|---|---|---|
| `install_dev.sh` | 从本仓库安装 vllm-plugin-FL + provenance 四重验证 | ⏳ |
| `run_baseline.sh` | 官方参数起 serve + 连通性 + 4K/16K 官方脚本双场景 + CSV 落 results/ | ⏳ |
| `run_eval.sh` | evalscope math_500 Level 3 精度 gate | ⏳ |
| `compare_results.py` | 台账 CSV 对比（基线 vs 当前，标注 ±1% 波动带） | ⏳ |
| `verify_provenance.sh` | pip 来源核验（独立入口，任何测量前调用） | ⏳ |

GPU 纪律：任何脚本起 serve/bench 前检查公告板（/workspace/p0-baseline/README）与 `pgrep -f vllm`；结束后 kill 并确认显存释放。
