#!/bin/bash
# sync_collab.sh — 双机协作同步检查（A/B 机共用；幂等；只读+提示，不自动 merge）
# 用法：bash scripts/d5/sync_collab.sh   （建议每 20-30 分钟一次，见 docs/远端C500引导.md §5）
# 输出：远端新提交 / 新远端分支 / 任务板与留言区变更 / 建议动作
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)/.."

CUR=$(git branch --show-current)
echo "== $(date '+%F %T') | $(hostname 2>/dev/null || echo '?') | 分支 $CUR@$(git rev-parse --short HEAD) =="
if ! git fetch origin --quiet 2>/dev/null; then
  echo "SYNC_FAIL: git fetch 失败（ghfast 单点故障？本地继续，恢复后重试）"
  exit 1
fi

UPD=0
# 1) 已有本地跟踪的分支：远端是否领先
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/ | sed 's|^origin/||' | grep -v HEAD); do
  LB=$(git rev-parse --verify -q "refs/heads/$b" >/dev/null && echo "$b" || echo "")
  BASE="${LB:-$(git rev-parse HEAD)}"
  N=$(git rev-list --count "$BASE..origin/$b" 2>/dev/null || echo 0)
  if [ "${N:-0}" -gt 0 ]; then
    UPD=1
    echo "-- 远端 $b 领先 ${N} 个提交："
    git log --oneline --no-decorate "$BASE..origin/$b" | head -5
    [ "$b" = "delivery/s2-final" ] && echo "   >> 建议：若本机无未提交工作，执行 git merge --ff-only origin/delivery/s2-final"
    [ -n "$LB" ] || echo "   >> 注意：本机无此分支（新分支），需要时 git checkout -b $b origin/$b"
  fi
done
# 2) 本机领先未 push 的（提醒自己 push）
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
  N=$(git rev-list --count "origin/$b..$b" 2>/dev/null || echo 0)
  [ "${N:-0}" -gt 0 ] && { UPD=1; echo "-- 本机 $b 领先远端 ${N} 个提交（记得 push）"; }
done

# 3) 任务板 / 留言区变更（远端 delivery/s2-final 上）
if [ "$(git rev-list --count HEAD..origin/delivery/s2-final 2>/dev/null || echo 0)" -gt 0 ]; then
  D=$(git diff HEAD origin/delivery/s2-final -- docs/双机任务板.md 2>/dev/null | grep -E '^[+-][^+-]' | head -30)
  [ -n "$D" ] && { echo "-- 任务板有变更："; echo "$D"; }
fi

[ "$UPD" -eq 0 ] && echo "NO_UPDATE（远端无新东西）"
exit 0
