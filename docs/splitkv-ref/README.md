# ⑥ 交接参考快照（vllm 0.24 site-packages 原文件，非本仓库代码）

- `vllm0.24_triton_unified_attention.py`：被优化的 attention kernel+wrapper
  （交接书 §4 的行号即对本文件）。
- `vllm0.24_triton_attn.py`：metadata builder（segm 缓冲/threshold，L55/L150-171）。
仅供无卡环境阅读对照；生产注入全部经插件 monkeypatch，这两个文件本仓库不改。
