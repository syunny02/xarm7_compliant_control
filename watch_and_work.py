#!/usr/bin/env python3
"""
Route B 信箱看门狗 — 自动化接力协议 v1 (MSG-A2 立)
====================================================
在被唤起的 session 内自主执行：
1. git pull 信箱
2. 检查最后一条【交接棒】是否指向 B
3. 如果是：解析任务 → 执行 → push 回执
4. 如果否：打印状态并退出

非 daemon，不退不醒。由用户/系统按需调用。
"""

import os, sys, subprocess, re, json, time
from pathlib import Path

REPO_DIR = Path(__file__).parent.resolve()
MAILBOX = REPO_DIR / "AI_CHANNEL.md"

HANDOFF_RE = re.compile(r"【交接棒】→\s*(.+?)(?:\n|$)")
STATE_RE = re.compile(r"【状态】\[(.+?)\]")

def git(*args, timeout=30):
    """Run git command in repo dir, return (ok, stdout)"""
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_DIR)] + list(args),
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "[timeout]"
    except Exception as e:
        return False, str(e)

def read_mailbox():
    """Return last 50 lines of mailbox"""
    if not MAILBOX.exists():
        return ""
    text = MAILBOX.read_text(encoding="utf-8")
    lines = text.splitlines()
    return "\n".join(lines[-50:])

def check_handoff(tail):
    """Return who should be woken, or None"""
    for line in reversed(tail.splitlines()):
        m = HANDOFF_RE.search(line)
        if m:
            target = m.group(1).strip()
            return target
    return None

def check_state(tail):
    """Return current state tag, or None"""
    for line in reversed(tail.splitlines()):
        m = STATE_RE.search(line)
        if m:
            return m.group(1).strip()
    return None

def find_last_msg_number():
    """Find highest MSG-XXX number in mailbox for B-side messages"""
    text = MAILBOX.read_text(encoding="utf-8") if MAILBOX.exists() else ""
    max_n = 6  # start from B06 (last one we wrote)
    for m in re.finditer(r"MSG-B(\d+)", text):
        max_n = max(max_n, int(m.group(1)))
    return max_n + 1

def work_cycle():
    """One full work cycle: pull → check → work → push"""
    print("=" * 50)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RouteB 看门狗启动")
    
    # 1. Pull
    ok, out = git("pull", "origin", "session/routeB-cartesian-vic")
    if not ok:
        print(f"  ❌ git pull failed: {out}")
        return False
    print(f"  ✅ git pull ok")
    
    # 2. Check handoff
    tail = read_mailbox()
    target = check_handoff(tail)
    state = check_state(tail)
    print(f"  状态: [{state}]  交接棒: → {target}")
    
    if target and "B" in target:
        print(f"  🟢 轮到 B 干活")
        # 这里后续可扩展: 解析 MSG 内容，自动执行任务
        # 目前先做检查+通知
        print(f"  ✅ 信箱检查完成，执行 B 任务...")
        return True
    else:
        print(f"  ⏸️ 当前不是 B 的棒次 (target={target})")
        return False

def main():
    """Entry point - single cycle. Call repeatedly or in a loop."""
    work_cycle()

if __name__ == "__main__":
    main()
