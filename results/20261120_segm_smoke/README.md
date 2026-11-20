# V150 P3 分段归因（2026-11-20）

**判决：P3 保持默认关闭。** 四臂的 FULL decode 图均捕获 16 段，服务期间观测到的 eager 调用也没有切到 8 段。阈值 0 全程无命中仍出现同类停顿。历史 82.3 秒对 43.0 秒的差距主要伴随请求在混批 prefill 和首 token 前等待，FlagGems 新形状初始化与停顿同窗出现。具体算子及耗时分摊仍缺直接计时证据。

## 实验口径

同一 BI-V150、同一时间窗口、同一端口 9035，依次接管阈值 1 的现有进程、重启阈值 0、SEGM off、阈值 8192。每次重启清空该 serve 的请求及 prefix cache 状态。各臂共用同一份 trace，并保留原有的磁盘编译和 FlagGems 调优缓存。请求由 64 个线程同时发起，实际混批形状受调度顺序影响。

固定环境为 `VLLM_PLUGINS=fl VLLM_FL_ATTN_BM=1 VLLM_FL_ATTN_TILE=0 VLLM_FL_ATTN_TD=1 P3_DIAG_TRACE=1`。实验变量为 `VLLM_FL_ATTN_SEGM` 和 `P3_DIAG_KV_LIMIT`。serve 参数如下。

```sh
vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9035 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

每臂等 `/health` 就绪后连续运行两次原脚本 `python3 -u scripts/d3/quick_smoke_client.py 9035 4096 64 128`。目标 ISL 4096 重分词为 4203 tokens。一次调用先发同提示的 24-token 流式单请求，再发 64 并发、128-token 非流式批次。因此首批也有 prefix cache，热批指同进程第二次调用。客户端 `总tok/s` 计入了缓存命中的提示。`/metrics` 在就绪、首轮结束、热轮结束各取一次，轮内指标用累计值之差并包含流式单请求。

本轮测试范围为上述非官方烟测。

## 对照结果

| 臂 | 首批秒 / 总 tok/s | 热批秒 / 总 tok/s | 平均 prefill 秒，首轮 / 热轮 | 平均 decode 秒，首轮 / 热轮 |
|---|---:|---:|---:|---:|
| SEGM=1，阈值 1 | 16.9 / 16358 | 56.4 / 4910 | 6.14 / 45.88 | 10.00 / 4.59 |
| SEGM=1，阈值 0 | 96.7 / 2866 | 49.7 / 5581 | 69.07 / 36.24 | 20.64 / 4.32 |
| SEGM=0 | 34.9 / 7938 | 20.1 / 13801 | 16.89 / 5.53 | 16.00 / 12.69 |
| SEGM=1，阈值 8192 | 75.1 / 3690 | 17.2 / 16151 | 48.60 / 5.43 | 21.39 / 9.99 |

8 个批次均为 64/64 成功。每轮平均值为 `request_{prefill,decode}_time_seconds` 的 `_sum` 增量除以 `_count` 增量。prefill 从请求获调度算到首 token，含该阶段的等待。decode 从首 token 算到末 token。各臂每轮均生成 8216 tokens，首轮和热轮分别有 98.20% 和 99.74% 的 prefix 命中。同一臂的热轮也可首次遇到某个混批形状。当前顺序和样本量适合判定各路径是否命中，以及定位停顿阶段。

## Warmup、Capture 与 Eager

- 安装的 vLLM 0.24.0 中，`gpu_model_runner.py::_warmup_and_capture` 先以 `CUDAGraphMode.NONE` 跑 dummy warmup，再以 `is_graph_capturing=True` 捕获。warmup 从 CPU seq len 计算 `max_seq_len=1`。实际 capture 取 `max_model_len=131072`，同时 `triton_attn.py::build_for_cudagraph_capture` 将设备上的 `seq_lens` 填成 1。hop1 的门控读取前者，即传给 wrapper 的 CPU 标量 `max_seqlen_k`。
- hook 打印按 `(capture, bs, q, kv, segments_before, segments_after)` 去重。进度条中的 `capture=0` 属于启动 warmup，HTTP 就绪之后的 `capture=0` 属于服务时的非图路径。FULL 图回放直接调用 `entry.cudagraph.replay()`，不再经过 Python hook。
- trace 中的 `segments` 是 wrapper 参数。模型有 2 个 KV heads，3D 上限为 `128 / 2 = 64`，且 64 为 capture size。`bs>64` 或 `q>1` 实际发射 2D，即使 hook 参数发生变化也没有 8 段 3D kernel。`bs=48/56/64, q=1` 的 dummy warmup 才是相关的 3D 对照。
- 实测每臂 35 个 `capture=1`，全为 `q=1,kv=131072,16->16`。阈值 1 和 8192 的 warmup 各有 27 次 `16->8` 参数命中，其中只有 `bs=48/56/64` 实际走 3D。阈值 0 和 off 的 warmup 全为 `16->16`。HTTP 就绪后每臂可见 7 个去重的 eager 形状，均为 `q>1,16->16`，其中并发批次的混批为 `q=11`。四臂的 graph capture 均在请求前完成，耗时 234 至 240 秒。

## 历史停顿

历史文件 `segm_on/quick.txt` 和 `segm_off/quick.txt` 分别记录 82.3 秒和 43.0 秒，均为 64/64 成功。两边的启动 capture 已在 HTTP 就绪前完成，耗时分别为 237 秒和 232 秒。旧 on 批次在 13:55:08 有 58 个运行中请求，生成速率只有 0.1 tok/s。13:55:18 时运行数仍为 58，prompt 和 generation 均为 0。13:55:48 时运行数升到 64，生成速率为 805.8 tok/s，随后 64 个请求完成。旧 off 在 14:02:56 已达到 64 个运行中请求和 524.8 tok/s，下一条统计显示全部完成。

旧 on 停顿两侧出现 FlagGems `resolve_benchmarker` 的 event-timing fallback 警告。警告来自 `LibTuner` 初始化时的 benchmarker 选择，能够标记冷初始化附近的活动，其本身没有提供算子名、编译时长或 autotune 时长。

同机 `/root/.flaggems/code_cache` 的文件 mtime 记录了若干混批形状的 `index_linearized` 初始化（北京时间）。

| 场景 | 文件 mtime | `index_linearized` 输入及索引桶 |
|---|---|---|
| 历史 on | 13:54:32.459 | BF16 `[628,2048]`，M 桶 64 |
| 历史 on | 13:55:04.330 | BF16 `[124,2048]`，M 桶 64 |
| 阈值 1 热轮 | 14:50:15.923 / 14:50:57.766 | BF16 `[606,2048]` / `[144,2048]`，M 桶 64 |
| 阈值 0 首轮 | 14:57:19.509 / 14:57:57.276 | BF16 `[584,2048]` / `[164,2048]`，M 桶 64 |
| off 首轮 | 15:06:07.644 / 15:06:19.592 | BF16 `[584,2048]` / `[164,2048]`，M 桶 64 |
| off 热轮 | 15:06:53.143 / 15:06:59.263 | BF16 `[606,2048]` / `[144,2048]`，M 桶 64 |
| 阈值 8192 首轮 | 15:13:18.602 / 15:13:36.838 | BF16 `[496,2048]` / `[244,2048]`，M 桶 64 |

FlagGems `ops/index.py::LinearizedAdjacentIndexFunction` 在新进程形状键首次触发时生成并载入对应文件。BF16 `[tokens,2048]` 的形状和 vLLM `gpu_model_runner.py` 模型前向后的 `hidden_states[logits_indices]` 一致。历史第二个形状 `124 = 58 + 6 × 11` 也与当时 58 路运行、再补齐到 64 路的混批相符。mtime 只记录代码写入时间，无法分摊后续 Python 初始化、JIT、autotune 或设备等待各自的耗时。

从阈值 0 首轮的混批轨迹可见，14:57:19 进入 `bs=54,q=11,kv=4204`，14:57:57 才进入 `bs=64,q=11,kv=4205`。14:58:08 服务仍报告 54 个运行中请求和 0 tok/s。两条 trace 都是 `capture=0, segments=16->16`。阈值 1 热轮在 `bs=56,q=11` 到 `bs=64,q=11` 之间也停了约 42 秒，prefill 均值从 6.14 秒涨到 45.88 秒，而 decode 均值下降。off 重遇这两对形状时，间隔分别约 12 秒和 6 秒。

阈值 8192 首轮在 15:13:47 有 46 个运行中请求和 0 tok/s，15:14:17 恢复到 64 个请求和 804.6 tok/s。与历史 on 一样，停顿后大量 token 在一个统计周期内产出。热批 17.2 秒说明该臂的首批 75.1 秒也不能视为稳定的 decode 速率。

## 归档文件

- `diag_*/serve.log.gz`、`first.txt` 和 `hot.txt` 是各臂的原始服务日志和客户端输出。`metrics_{before,first,hot}.prom.gz` 是原始指标快照。
- `segm_{on,off}/serve.log.gz` 和 `quick.txt` 是历史对照原件。
- `p3-trace.patch.gz` 保存了接手时已加载的诊断补丁。实验前及每次重启前比对仓库和 site-packages 源文件，两份诊断代码一致。复跑时在交接基线 `7574909` 应用补丁，并同步到所用 Python 的 `site-packages/vllm_fl/dispatch/backends/vendor/iluvatar/attn_launch_config.py`。

15:15:11 `ixsmi` 显示 68 MiB、GPU 利用率 0%、无运行进程。各臂退出均针对已核验监听 9035 的 serve PID，EngineCore 随其退出。复核 9035 已停止监听，两份插件源文件仍一致。
