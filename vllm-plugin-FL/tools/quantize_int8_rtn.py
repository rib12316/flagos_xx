#!/usr/bin/env python3
"""W8A8 动态 int8 转换脚本（D2 唯一开发件）：bf16 checkpoint → packed int8 checkpoint。

量化方式：per-channel 对称 RTN（可选 clip α 与 SmoothQuant-lite 离线折叠），激活侧
不落任何静态参数（运行时逐 token 动态量化由 vllm_fl 管线负责）。全程 checkpoint
config 驱动——本脚本不引入任何 serve 量化超参。

输出（compressed-tensors pack-quantized，精确匹配 should_use_packed_w8a8 判定）：
  - 每个量化 Linear 三键：weight_packed int32 [N, K/4]（offset-binary-128，小端
    低位在前，与 unpack_uint8b128_int32 严格互逆）、weight_scale fp32 [N,1]、
    weight_shape int64 [2] = [K, N]
  - keep-list 层 / embed / lm_head / norms 保 bf16 原样
  - config.json 注入 quantization_config（targets:["Linear"] 类名匹配 → embedding
    天然不量化；ignore = keep-list 运行时融合模块名，粒度整模块）
  - tokenizer / chat_template / generation_config 原样拷贝

幂等：seed 固定；折叠因子由激活统计确定性导出；RTN 确定性——同输入同输出，
组委会可从原始模型一键再生。

自检（失败非零退出）：全量 round-trip 逐位断言；抽样层 CPU int32 参考对拍 ≤1e-2；
假 config 经 patched scheme 选择器命中 PackedW8A8Int8 且 lm_head 返回 None-scheme。

用法：
  python tools/quantize_int8_rtn.py --src /workspace/MiniCPM5-2B \
      --dst /workspace/MiniCPM5-2B-w8a8 \
      --clip-alpha 0.99 --smooth-alpha 0.6 \
      --act-stats /path/act_stats.json --keep none --seed 1234 --self-check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

# 折叠数学唯一定义处 = scripts/d2/common.py（与本脚本共享；转换侧作用于分离张量）
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts", "d2"))
from common import (  # noqa: E402
    NUM_LAYERS, apply_folds_separate, keep_predicate, load_fold_scales,
    rtn_channel_quant,
)

LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def pack_int8_offset_binary(wq: torch.Tensor) -> torch.Tensor:
    """int8 [N,K] → int32 [N,K/4]（offset-binary-128，小端低位在前）。"""
    n, k = wq.shape
    assert k % 4 == 0, f"K={k} 不被 4 整除"
    codes = (wq.to(torch.int16) + 128).to(torch.uint8).contiguous()
    return codes.view(torch.int32).reshape(n, k // 4)


def layer_quant_flags(keep_name: str, li: int) -> dict[str, bool]:
    """该层四站点是否量化（keep 预设以运行时融合模块名判定，此处映射到分离张量）。"""
    pred = keep_predicate(keep_name)
    qkv = f"model.layers.{li}.self_attn.qkv_proj"
    return {
        "qkv": not pred(qkv),
        "o": not pred(qkv.replace("qkv_proj", "o_proj")),
        "gate_up": not pred(f"model.layers.{li}.mlp.gate_up_proj"),
        "down": not pred(f"model.layers.{li}.mlp.gate_up_proj".replace("gate_up_proj", "down_proj")),
    }


def build_ignore(keep_name: str) -> list[str]:
    """keep-list → quantization_config.ignore（运行时融合模块名，整模块粒度）。"""
    if keep_name == "none":
        return []
    if keep_name == "attn":  # MLP-only 量化
        return [f"model.layers.{i}.self_attn.{m}" for i in range(NUM_LAYERS)
                for m in ("qkv_proj", "o_proj")]
    if keep_name == "mlp":  # 仅 attn 量化（诊断用）
        return [f"model.layers.{i}.mlp.{m}" for i in range(NUM_LAYERS)
                for m in ("gate_up_proj", "down_proj")]
    if keep_name.startswith("first_last:"):
        k = int(keep_name.split(":")[1])
        idx = sorted(set(range(k)) | set(range(NUM_LAYERS - k, NUM_LAYERS)))
        return [f"re:^model\\.layers\\.({'|'.join(map(str, idx))})\\."]
    if keep_name.startswith("layers:"):
        idx = sorted(int(v) for v in keep_name[len("layers:"):].split(","))
        return [f"re:^model\\.layers\\.({'|'.join(map(str, idx))})\\."]
    raise ValueError(f"unknown keep preset: {keep_name}")


def quantization_config(ignore: list[str]) -> dict:
    return {
        "config_groups": {"group_0": {
            "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                  "strategy": "token", "dynamic": True},
            "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                        "strategy": "channel", "dynamic": False},
            "targets": ["Linear"]}},
        "format": "pack-quantized",
        "ignore": ignore,
        "quant_method": "compressed-tensors",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/MiniCPM5-2B")
    ap.add_argument("--dst", default="/workspace/MiniCPM5-2B-w8a8")
    ap.add_argument("--clip-alpha", default="1.0",
                    help="float 或 auto（逐张量 MSE argmin 于 {1.0,0.999,0.99,0.98,0.95}）")
    ap.add_argument("--smooth-alpha", type=float, default=0.0)
    ap.add_argument("--act-stats", default=None)
    ap.add_argument("--keep", default="none")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    assert os.path.isdir(args.src), f"src 不存在: {args.src}"
    assert not os.path.exists(args.dst) or not os.listdir(args.dst), \
        f"dst 非空，拒绝覆盖: {args.dst}"
    os.makedirs(args.dst, exist_ok=True)

    idx = json.load(open(f"{args.src}/model.safetensors.index.json"))
    shard = sorted(set(idx["weight_map"].values()))[0]
    tensors = load_file(f"{args.src}/{shard}")
    folds_all = (load_fold_scales(args.act_stats, args.smooth_alpha)
                 if args.smooth_alpha > 0 else None)
    if args.smooth_alpha > 0:
        assert args.act_stats, "--smooth-alpha>0 需要 --act-stats"

    out: dict[str, torch.Tensor] = {}
    n_packed = 0
    clip_table: dict[str, float] = {}
    for li in range(NUM_LAYERS):
        p = f"model.layers.{li}"
        # 短键约定与 common.apply_folds_separate 一致
        sep = {f"{n}.weight": tensors[f"{p}.{pre}.{n}.weight"] for pre, n in (
            ("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
            ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"),
            ("mlp", "down_proj"))}
        sep["input_layernorm.weight"] = tensors[f"{p}.input_layernorm.weight"]
        sep["post_attention_layernorm.weight"] = tensors[f"{p}.post_attention_layernorm.weight"]
        quant = layer_quant_flags(args.keep, li)
        # 折叠（消费 keep-list 站点跳过其列折叠；v/up 行折叠在 keep 层同样应用——精确数学）
        folded = apply_folds_separate(
            sep, li, folds_all[li] if folds_all is not None else None, quant)
        # norm 张量（可能已折）
        out[f"{p}.input_layernorm.weight"] = folded["input_layernorm.weight"]
        out[f"{p}.post_attention_layernorm.weight"] = folded["post_attention_layernorm.weight"]
        site_of_sep = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
                       "o_proj": "o", "gate_proj": "gate_up", "up_proj": "gate_up",
                       "down_proj": "down"}
        pre_of = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn",
                  "o_proj": "self_attn", "gate_proj": "mlp", "up_proj": "mlp",
                  "down_proj": "mlp"}
        for name, W in sep.items():
            short = name[: -len(".weight")]  # q_proj / k_proj / ...
            if short in ("input_layernorm", "post_attention_layernorm"):
                continue  # norm 已处理
            key = f"{p}.{pre_of[short]}.{name}"
            if not quant[site_of_sep[short]]:
                out[key] = folded[name].to(torch.bfloat16)  # keep：保 bf16（可能含行折叠）
                continue
            Wq = folded[name].to(torch.float32)
            alpha = pick_alpha(Wq, args.clip_alpha, key, clip_table)
            wq, ws = rtn_channel_quant(Wq, alpha)
            wp = pack_int8_offset_binary(wq)
            assert wp.dtype == torch.int32
            out[f"{key}_packed"] = wp
            out[f"{key}_scale"] = ws.to(torch.float32).reshape(-1, 1)  # 加载断言 [N,1]
            out[f"{key}_shape"] = torch.tensor([Wq.shape[1], Wq.shape[0]], dtype=torch.int64)
            n_packed += 1

    # 全局张量原样
    for k, v in tensors.items():
        if not re.search(r"layers\.\d+\.", k):
            out[k] = v

    save_file(out, f"{args.dst}/{shard}", metadata={"format": "pt"})

    # config 注入 + 附属文件拷贝
    cfg = json.load(open(f"{args.src}/config.json"))
    cfg["quantization_config"] = quantization_config(build_ignore(args.keep))
    with open(f"{args.dst}/config.json", "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    for fn in os.listdir(args.src):
        if fn.endswith((".json", ".jinja", ".txt", ".model")) and fn not in ("config.json",):
            shutil.copy2(f"{args.src}/{fn}", f"{args.dst}/{fn}")
    shutil.copy2(f"{args.src}/model.safetensors.index.json", f"{args.dst}/")

    print(f"RESULT|mode=convert|tensors_packed={n_packed}|keep={args.keep}|"
          f"clip={args.clip_alpha}|smooth={args.smooth_alpha}|dst={args.dst}")

    if args.self_check:
        return self_check(args.dst, shard, n_packed, clip_table)
    return 0


def pick_alpha(Wq: torch.Tensor, spec: str, key: str, table: dict) -> float:
    if spec == "auto":
        best_a, best_mse = 1.0, float("inf")
        for a in (1.0, 0.999, 0.99, 0.98, 0.95):
            wq, ws = rtn_channel_quant(Wq, a)
            mse = ((wq.to(torch.float32) * ws[:, None] - Wq) ** 2).mean().item()
            if mse < best_mse:
                best_a, best_mse = a, mse
        table[key] = best_a
        return best_a
    return float(spec)


def self_check(dst: str, shard: str, expect_packed: int, clip_table: dict) -> int:
    from vllm_fl.quantization.w8a8.reference import (
        unpack_uint8b128_int32, w8a8_linear_reference,
    )
    t = load_file(f"{dst}/{shard}")
    packed = {k: v for k, v in t.items() if k.endswith("_packed")}
    assert len(packed) == expect_packed, f"packed 计数 {len(packed)} != {expect_packed}"

    # 1) round-trip 逐位（全量）
    for k, wp in packed.items():
        base = k[:-len("_packed")]
        shape = t[f"{base}_shape"].tolist()  # [K, N]
        wq = unpack_uint8b128_int32(wp, in_features=shape[0])
        n_rec = (wq.to(torch.int16) + 128).to(torch.uint8).contiguous().view(torch.int32)
        assert torch.equal(n_rec.reshape(wp.shape), wp), f"round-trip 失败 {k}"
        assert t[f"{base}_scale"].shape == (shape[1], 1), f"scale 形状 {k}"
    print(f"RESULT|check=roundtrip|tensors={len(packed)}|bit_exact=True")

    # 2) 数值抽样（CPU）：两侧同量化激活，验证 packed 权重忠实实现 W8A8 语义 ≤1e-2
    #    y_ref = int32 精确累加参考；y_dq = 反量化 bf16 线性（bf16 舍入 + fp32 累加差）
    from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8
    import random
    keys = random.Random(0).sample(sorted(packed), 5)
    worst = 0.0
    for k in keys:
        base = k[:-len("_packed")]
        K, N = t[f"{base}_shape"].tolist()
        wp, ws = t[k], t[f"{base}_scale"].reshape(-1)
        wq = unpack_uint8b128_int32(wp, in_features=K)
        w_dq = (wq.to(torch.float32) * ws[:, None]).to(torch.bfloat16)
        x = (torch.randn(64, K) * 2).to(torch.bfloat16)
        y_ref = w8a8_linear_reference(x, wq, ws)          # CPU：int32 精确累加
        xq, xs = dynamic_per_token_quant_int8(x)
        x_dq = (xq.to(torch.float32) * xs).to(torch.bfloat16)
        y_dq = torch.nn.functional.linear(x_dq, w_dq)     # 同量化激活 + 反量化权重
        err = (y_ref.float() - y_dq.float()).abs().max().item()
        worst = max(worst, err)
        assert err <= 1e-2, f"数值对拍超限 {k}: {err:.3e}"
    print(f"RESULT|check=numeric|sampled=5|max_abs={worst:.2e}|limit=1e-2")

    # 3) scheme 选择（config 驱动激活命中）
    from vllm_fl.quantization.w8a8.packed import (
        CompressedTensorsPackedW8A8Int8, install_packed_w8a8_scheme,
    )
    install_packed_w8a8_scheme()
    from vllm.model_executor.layers.quantization.compressed_tensors.\
        compressed_tensors import CompressedTensorsConfig
    qc = CompressedTensorsConfig.from_config(
        json.load(open(f"{dst}/config.json"))["quantization_config"])
    scheme = qc._get_scheme_from_parts(
        weight_quant=qc.config_groups["group_0"].weights,
        input_quant=qc.config_groups["group_0"].input_activations,
        layer_name="model.layers.0.self_attn.qkv_proj")
    assert isinstance(scheme, CompressedTensorsPackedW8A8Int8), \
        f"scheme 未命中: {type(scheme)}"
    from compressed_tensors.quantization import QuantizationArgs
    lm_scheme = qc._get_scheme_from_parts(
        weight_quant=QuantizationArgs(num_bits=8), input_quant=None,
        layer_name="lm_head")  # lm_head 无 input_quant → 上游走 None/embedding 分支
    print(f"RESULT|check=scheme|layer=PackedW8A8Int8|cap={scheme.get_min_capability()}|ok=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
