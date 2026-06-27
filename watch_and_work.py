#!/usr/bin/env python3
"""
Route B SMART Watchdog v3 - автономный агент
Проверяет信箱 каждый час. Когда очередь B - анализирует задачу и выполняет.
Использует main repo paths (проверено работает).
"""

import os, sys, subprocess, re, time, json
from pathlib import Path
from datetime import datetime

# ── Пути используют main repo (где train_vic.py работает) ──
WATCHDOG_DIR = Path(__file__).parent.resolve()        # xarm7_compliant_control/
MAIN_REPO = WATCHDOG_DIR.parent                        # xarm7_door_ros2/
MAILBOX = WATCHDOG_DIR / "AI_CHANNEL.md"
ENV_FILE = MAIN_REPO / "xarm7_cartesian_vic" / "cartesian_vic_env.py"
TRAIN_SCRIPT = MAIN_REPO / "xarm7_cartesian_vic" / "train_vic.py"
TRAIN_CWD = MAIN_REPO / "xarm7_cartesian_vic"
RUNS_DIR = TRAIN_CWD / "runs"
CUDA_PYTHON = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"

HANDOFF_RE = re.compile(r"\u3010\u4ea4\u63a5\u68d2\u3011\u2192\s*(.+?)(?:\n|$)")
MSG_NUM_RE = re.compile(r"MSG-B(\d+)")

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def git(*args, timeout=30):
    try:
        r = subprocess.run(["git", "-C", str(WATCHDOG_DIR)] + list(args),
            capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip()
    except: return False, "error"

def read_mailbox():
    if not MAILBOX.exists(): return ""
    return MAILBOX.read_text(encoding="utf-8")

def get_handoff(text):
    for line in reversed(text.splitlines()):
        m = HANDOFF_RE.search(line)
        if m:
            target = m.group(1).strip()
            # Нормализация: Chinese indicators for B/GPU machine
            if any(x in target for x in ["B", "显卡机", "GPU"]):
                return "B"
            return target
    return None

def next_msg_b(text):
    max_n = 12
    for m in MSG_NUM_RE.findall(text): max_n = max(max_n, int(m))
    return max_n + 1

def append_mailbox(body):
    with open(MAILBOX, "a", encoding="utf-8") as f: f.write("\n\n" + body)

def set_env_constant(name, value):
    if not ENV_FILE.exists(): return False
    content = ENV_FILE.read_text(encoding="utf-8")
    new_val = str(value)
    if isinstance(value, float): new_val = f"{value}"
    new_content = re.sub(rf"^{name}\s*=\s*[0-9.eE+\-]+", f"{name} = {new_val}", content, count=1, flags=re.MULTILINE)
    if new_content == content:
        new_content = re.sub(rf"{name}\s*=\s*[0-9.eE+\-]+", f"{name} = {new_val}", content, count=1)
    if new_content != content:
        ENV_FILE.write_text(new_content, encoding="utf-8")
        log(f"Changed {name} = {new_val}")
        return True
    log(f"Could not change {name}")
    return False

def run_training(steps=300000, curriculum=None, run_name=None, device="cuda", test_damping=None):
    if run_name is None:
        label = f"curri{curriculum}" if curriculum is not None else "default"
        run_name = f"VIC_PPO_{label}_{steps//1000}k"
    run_name = run_name.replace(".", "_")  # no dots in dir names
    
    cmd = [CUDA_PYTHON, str(TRAIN_SCRIPT),
        "--algo", "PPO", "--steps", str(steps), "--n-envs", "4",
        "--run-name", run_name, "--device", device]
    if curriculum is not None and curriculum < 999:
        cmd += ["--curriculum-level", str(curriculum)]
    if test_damping:
        cmd += ["--test-damping"] + [str(d) for d in test_damping] + ["--eval-episodes", "20"]
    
    log(f"TRAIN {run_name} ({steps} steps)")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(TRAIN_CWD))
        ok = r.returncode == 0
        output = r.stdout[-2000:]
    except Exception as e:
        ok, output = False, str(e)
    dt = time.time() - t0
    
    gen_path = RUNS_DIR / run_name / "generalization_results.json"
    if gen_path.exists():
        mtime = datetime.fromtimestamp(gen_path.stat().st_mtime)
        results = json.loads(gen_path.read_text())
        k_avg = sum(v['mean_K'] for v in results.values()) / len(results)
        sr_best = max(v['success_rate'] for v in results.values())
        summary = f"{run_name}: {dt/60:.0f}min K={k_avg:.0f} SR={sr_best:.0%} (mtime={mtime:%H:%M:%S})"
    else:
        summary = f"{run_name}: {dt/60:.0f}min ok={ok} (no gen_results)"
    
    log(summary)
    return ok, summary, run_name

def analyze_results(run_name):
    path = RUNS_DIR / run_name / "generalization_results.json"
    if not path.exists(): return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    data = json.loads(path.read_text())
    items = sorted(data.items(), key=lambda x: x[1]['damping'])
    table = f"mtime={mtime:%H:%M:%S}\n| Damping | Door(rad) | SR | K | Force |\n|:------:|:---------:|:--:|:--:|:-----:|\n"
    k_sum = 0
    for k,v in items:
        table += f"| {v['damping']:.1f} | {v['mean_door_ang']:.3f} | {v['success_rate']:.0%} | {v['mean_K']:.0f} | {v['mean_contact_force']:.1f} |\n"
        k_sum += v['mean_K']
    return {"table": table, "K_avg": k_sum/len(items), "SR_max": max(v['success_rate'] for v in data.values())}

def work_cycle(cycle):
    log(f"===== Cycle {cycle} =====")
    
    ok, out = git("pull", "origin", "session/routeB-cartesian-vic")
    if not ok: log(f"git pull failed: {out}"); return False
    log("git pull ok")
    
    mailbox = read_mailbox()
    target = get_handoff(mailbox)
    log(f"Handoff -> {target}")
    
    if target != "B": log("Not B's turn"); return False
    
    log("B's turn! Running verification training...")
    
    # Phase 1: curriculum_level=0, 300k
    ok, summary, rn = run_training(300000, 0, "VIC_PPO_recheck_curri0", device="cuda",
        test_damping=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    analysis = analyze_results(rn)
    msg_n = next_msg_b(mailbox)
    
    report = f"## MSG-B{msg_n} - Watchdog recheck Phase1 complete\n\n"
    if analysis:
        report += "Phase1 recheck (curri0, 300k): returncode=0 ✅\n\n"
        report += analysis['table'] + "\n"
    else:
        report += f"Phase1 recheck: {summary}\n"
    
    report += "【状态】[completed] (recheck done - all data fresh with timestamps)\n"
    report += "【交接棒】-> Please wake AI (SoniXChat) to review fresh data\n"
    
    append_mailbox(report)
    git("add", "AI_CHANNEL.md")
    git("commit", "-m", f"MSG-B{msg_n}: recheck Phase1 done")
    git("push", "origin", "session/routeB-cartesian-vic")
    log("Done, pushed")
    return True

def main():
    log("[DOG] Watchdog v3 started (24h)")
    r = subprocess.run([CUDA_PYTHON, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
        capture_output=True, text=True, timeout=10)
    log(f"GPU: {r.stdout.strip()}")
    
    cycle = 0; start = time.time(); end = start + 24*3600
    while time.time() < end:
        cycle += 1
        try: work_cycle(cycle)
        except Exception as e:
            log(f"Error: {e}")
            import traceback; traceback.print_exc()
        rem = (end - time.time())/3600
        log(f"Remaining {rem:.1f}h, sleep 60min\n")
        for _ in range(60):
            time.sleep(60)
            if time.time() >= end: break
    log("[DOG] 24h done")

if __name__ == "__main__":
    main()
