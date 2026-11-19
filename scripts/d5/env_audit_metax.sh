#!/bin/bash
# env_audit_metax.sh — 沐曦 C500 环境只读审计（B 机开工硬门；A 机复用）
# 判定：任何 FAIL → 退出码 1，不开工。WARN → 记录在案再继续。
# 用法：bash scripts/d5/env_audit_metax.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

PASS=0; FAIL=0; WARN=0
ok()   { echo "PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "WARN  $*"; WARN=$((WARN+1)); }

echo "== 1. GPU / 驱动 =="
if command -v mx-smi >/dev/null 2>&1; then
  ok "mx-smi 存在 ($(mx-smi --version 2>/dev/null | head -1))"
  MXOUT=$(mx-smi 2>/dev/null)
  echo "$MXOUT" | grep -q 'MACA Version: 3.7.0.38' && ok "MACA 3.7.0.38" || warn "MACA 版本非 3.7.0.38（对照 A 机：3.7.0.38）"
  echo "$MXOUT" | grep -q 'no process found' && ok "GPU 空闲（无进程）" || warn "GPU 上有进程（bench/测量前必须空闲）"
  USED=$(echo "$MXOUT" | grep -oE '[0-9]+/65536 MiB' | head -1 | cut -d/ -f1)
  [ "${USED:-99999}" -lt 2048 ] && ok "显存占用 ${USED}MiB < 2GB" || warn "显存占用 ${USED:-?}MiB 偏高"
else
  bad "mx-smi 不存在——不是沐曦环境？"
fi
pgrep -af 'vllm serve' && warn "存在 vllm serve 残留（官方 run 前必须清掉）" || ok "无 vllm serve 残留"

echo "== 2. Python 栈版本 =="
PY=python3
V=$($PY -c "import torch;print(torch.__version__)" 2>/dev/null) || bad "torch 不可导入"
[ "${V:-}" = "2.8.0+metax3.7.0.7" ] && ok "torch $V" || { [ -n "${V:-}" ] && warn "torch $V ≠ A机 2.8.0+metax3.7.0.7"; }
V=$($PY -c "import triton;print(triton.__version__)" 2>/dev/null) || bad "triton 不可导入"
[ "${V:-}" = "3.0.0+metax3.7.0.7" ] && ok "triton $V" || warn "triton $V ≠ A机 3.0.0+metax3.7.0.7"
V=$($PY -c "import vllm;print(vllm.__version__)" 2>/dev/null) || bad "vllm 不可导入"
[ "${V:-}" = "0.24.0+empty" ] && ok "vllm $V" || warn "vllm $V ≠ 0.24.0+empty"
V=$($PY -c "import flash_attn;print(flash_attn.__version__)" 2>/dev/null) || bad "flash_attn 不可导入"
echo "${V:-}" | grep -q '2.6.3+metax' && ok "flash_attn $V" || warn "flash_attn ${V:-?}（A机 2.6.3+metax3.7.0.7torch2.8）"

echo "== 3. 本仓插件安装与 provenance =="
LOC=$($PY -c "import vllm_fl,os;print(os.path.dirname(vllm_fl.__file__))" 2>/dev/null) || bad "vllm_fl 不可导入"
if [ -n "${LOC:-}" ]; then
  ok "vllm_fl → $LOC"
  SP=$(echo "$LOC" | grep -oE '^.*/site-packages')
  DU="$SP/vllm_plugin_fl-0.0.0.dist-info/direct_url.json"
  if [ -f "$DU" ]; then
    grep -q "flagos_xx" "$DU" && ok "direct_url.json 指向本仓克隆" || warn "direct_url.json 非本仓：$(cat "$DU" | head -c 120)"
  else
    # editable/开发装也可能没有该文件
    warn "无 $DU（若为 editable 安装可接受，须确认指向本仓）"
  fi
  diff -rq "$LOC" ./vllm-plugin-FL/vllm_fl >/dev/null 2>&1 && ok "site-packages 与仓库源码一致" || warn "site-packages 与仓库源码有差异（重装：pip install --no-deps --no-build-isolation ./vllm-plugin-FL）"
fi
V=$(pip show flag_gems 2>/dev/null | awk '/^Version/{print $2}')
[ "${V:-}" = "5.3.5" ] && ok "flag_gems $V" || bad "flag_gems ${V:-缺失} ≠ 5.3.5"

echo "== 4. mctlassEx（w8a8 mctlass 承载，缺失=静默退化风险点）=="
$PY -c "import torch,mctlassEx;print('ok')" >/dev/null 2>&1 && ok "import torch; import mctlassEx 成功" || bad "mctlassEx 导入失败（镜像预装件缺失 → w8a8 会退回慢路径）"

echo "== 5. 权重与数据 =="
for D in /workspace/MiniCPM5-2B /workspace/MiniCPM5-2B-w8a8-v2; do
  if [ -f "$D/config.json" ]; then
    ok "$D 存在"
  else
    bad "$D 缺失" $( [ "$D" = "/workspace/MiniCPM5-2B" ] && echo "（需下载 openbmb/MiniCPM5-2B，见 docs/远端C500引导.md）" || echo "（用 quantize_int8_rtn.py 再生，见 DELIVERY.md）" )
  fi
done
$PY - <<'EOF' >/dev/null 2>&1 && ok "w8a8 ckpt 含 quantization_config" || warn "w8a8 ckpt quantization_config 检查未过（或 ckpt 未再生）"
import json;c=json.load(open('/workspace/MiniCPM5-2B-w8a8-v2/config.json'));assert 'quantization_config' in c
EOF
[ -f results/20261107_d2_actstats/act_stats.json ] && ok "act_stats.json 在仓" || warn "act_stats.json 缺失（w8a8 再生需要）"
$PY -c "import evalscope;print(evalscope.__version__)" 2>/dev/null | grep -q '^1\.11\.' && ok "evalscope 1.11.x" || warn "evalscope 非 1.11.x（精度评测口径用）"

echo "== 6. git / 网络 =="
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && ok "git 仓库：$(git branch --show-current)@$(git rev-parse --short HEAD)" || bad "不在 git 仓库"
git ls-remote origin HEAD >/dev/null 2>&1 && ok "远端可达（fetch 走 ghfast 镜像）" || bad "远端不可达（检查 ghfast.top 或 SSH key）"

echo
echo "== 汇总: PASS=$PASS FAIL=$FAIL WARN=$WARN =="
[ "$FAIL" -eq 0 ] && echo "AUDIT_GREEN（WARN 项记录后可开工）" || { echo "AUDIT_RED——修复 FAIL 后再开工"; exit 1; }
