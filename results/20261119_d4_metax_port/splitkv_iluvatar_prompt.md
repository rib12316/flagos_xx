# ⑥ split-KV decode kernel 天数侧实测调优 · agent prompt

> 生成于 2026-11-19（沐曦机 D4 会话）。目标机：**天数 BI-V150 实例**（16 SM，
> triton 3.6 flagtree iluvatar backend）。截止：2026-11-20 23:59，机时窗口紧。

## 给 agent 的 prompt（直接粘贴）

```
你在天数 BI-V150 实例上执行 FlagOS 比赛仓库 /data/flagos_xx（或你机器上的
clone）的 ⑥ split-KV decode kernel 实测调优任务。该 kernel 由无卡 agent 于
11-19 交付（commit 06a2ef5），代码零实测——你是第一个接触 GPU 的人。

# 分支与权威文档（先读，别跳）
- 分支 feat/d2-w8a8-quant @06a2ef5（GitHub 已有）；若拿到 d4-metax-port.bundle
  则 git fetch <bundle> feat/d4-metax-port（多 2 个沐曦惰性 commit，不改天数行为）
- docs/splitkv-kernel-handoff.md ——唯一权威交接书：§7A 无卡风险5条+预案、
  §7B 真机debug预案表A1-E1、§8 六步真机计划（时间盒）、§13.4 修订版次日计划
- docs/improvements-master.md ——改进总账+全部 env 门控+回退矩阵
- tests/unit_tests/attention/test_splitkv_decode.py 文件头——真机 T0 命令序列
- 对照基线（天数 B 臂终局）：4K 3414.58 / 16K 1570.62 / 105题 0.9714
  （B 臂 = w8a8-v2 ckpt + VLLM_FL_ATTN_TILE=1 VLLM_FL_ATTN_TD=1）

# 纪律（硬约束）
- pip 只从本仓库 vllm-plugin-FL 安装（--no-deps --no-build-isolation），装后
  provenance 四查（location/direct_url/site-packages diff/flag_gems==5.3.5）
- serve 参数一字不动（§8 红线命令）；bench 用仓库原件 benchmark_throughput_serve.py
- 单测/probe 异常现场调参 ≤30min，超时按 §7B 预案降级（V1 降 GROUP 3→2→1；
  V3 弃；V0 兜底），数据结案不留恋
- 每步产物落 results/ 目录 + DEVLOG 回写；EngineCore 死亡（27-80min 随机时钟）
  = 换 serve 重跑该场景，勿长跑串联

# 六步计划（§8，按序，含 go/no-go）
1. T0 编译/装载自检（15min）：单测 test_splitkv_decode.py -x -q；编译失败按
   §7B-A1 处理，失败即修，不动下一步
2. T1 kernel 微基准（20min）：python3 scripts/d2/attn_surface_probe.py
   --suite splitkv --out-dir results/d3_dev/item6_splitkv
   门槛：官方形态（bs32-40/kv16.4-17.4K 平均）≥1.15×（<1520µs@bs32 口径）
   才继续；未达 → 数据结案收工（不要现场调参超 30min）
3. T2 serve 集成烟测（45min）：B 臂 env + VLLM_FL_ATTN_SPLITKV=1 起 serve
   （红线命令见 §8）→ serve.log grep "fl.attn-splitkv: routed" 确认路由
   （主进程+EngineCore 双进程装载，两行）→ python3 scripts/d3/parity_check.py
   compare（对照 results/d3_dev/parity_baseline.json；⑥ 非位精确，<16/16 记录
   分歧率）→ decode 代理 bench --test-cases '[[256,1024,64,64]]'（ITL 对照
   B 态 ~16.2ms）
4. T3 官方 16K 单轮试金石（35min）：新鲜 serve + 单轮 [[16384,1024,64,128]]；
   对照 B16 轮值 1358/1573/1569/1570；≥+5% 才值得 T4
5. T4 官方 16K 完整 4 轮（~2h）：scripts/d3/run_official_redo.sh B16 改 env
   或手动等价；每场景新鲜 serve
6. T5 105 题精度门（45min）：evalscope math_500 L3，红线 ≥0.95（B 态 0.9714）；
   过线 → 回写 docs/improvements-master.md §0 与交付包数字

# serve 红线命令（一字不动，只加 env）
VLLM_PLUGINS=fl VLLM_FL_ATTN_TILE=1 VLLM_FL_ATTN_TD=1 VLLM_FL_ATTN_SPLITKV=1 \
vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072

# 已知背景（避免重复踩坑）
- kernel 严禁 libentry/atomic/autotune（纪律有前科）；V1 双缓冲不用 num_stages
- 自管 segm 缓冲按 token 索引（修上游 #48076 per-seq 分配根因）；capture 期
  静态分段=cudagraph 安全；工作缓冲只增长不搬迁（graph 安全），日志见
  "工作缓冲增长"属正常（capture 前完成即可）
- ①TD 与 ⑥ 的 TD 模式互作用：VLLM_FL_ATTN_SPLITKV_TD 保持默认（B 臂 TD=1
  时 kernel 内 use_td 由路由传入，本 hop 恒 False 的设定仅沐曦）
- 沐曦侧今日战况（供报告参考，不影响你）：C500 上 w8a8 经 mctlassEx 复活，
  4K 首轮 5804 tok/s；⑥ 的沐曦 hop 已写好（vendor/metax/attn_splitkv_hop.py，
  env 同名门控），沐曦基线官方数字今晚出
```

## 沐曦侧随附信息（非任务必需）

- `d4-metax-port.bundle`（68MB）含全部 50 commits；天数机无需它也能干活
  （feat/d2-w8a8-quant 已在 GitHub）
- 沐曦今日已证：triton 3.0 栈能编译 vllm core triton kernel（topk_topp 探针
  过）——天数 3.6 flagtree 编译 ⑥ 大概率无碍，但 T0 仍最先验证
