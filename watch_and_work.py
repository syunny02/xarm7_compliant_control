#!/usr/bin/env python3
"""
Route B mailbox watchdog - GPU machine auto-loop
Check mailbox every 60 min, execute B-tasks, update and push.
Use Python 3.12 (cu128 torch) for GPU training.
"""

import os, sys, subprocess, re, time, json
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.resolve()
MAIN_REPO = REPO_DIR.parent  # xarm7_door_ros2
MAILBOX = REPO_DIR / "AI_CHANNEL.md"
TRAIN_SCRIPT = MAIN_REPO / "xarm7_cartesian_vic" / "train_vic.py"  # main repo's proven script
TRAIN_SCRIPT_DIR = MAIN_REPO / "xarm7_cartesian_vic"  # working directory for training
RUNS_DIR = MAIN_REPO / "xarm7_cartesian_vic" / "runs"

HANDOFF_RE = re.compile(r"\u3010\u4ea4\u63a5\u68d2\u3011\u2192\s*(.+?)(?:\n|$)")
STATE_RE = re.compile(r"\u3010\u72b6\u6001\u3011\[(.+?)\]")
MSG_NUM_RE = re.compile(r"MSG-B(\d+)")

CUDA_PYTHON = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def git(*args, timeout=60):
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_DIR)] + list(args),
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0, r.stdout.strip()
    except Exception as e:
        return False, str(e)

def run_cmd(cmd, timeout=7200):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout[-3000:] + r.stderr[-1000:]
    except subprocess.TimeoutExpired:
        return -1, "[timeout]"
    except Exception as e:
        return -1, str(e)

def read_tail(lines=80):
    if not MAILBOX.exists():
        return ""
    text = MAILBOX.read_text(encoding="utf-8")
    return "\n".join(text.splitlines()[-lines:])

def read_full():
    if not MAILBOX.exists():
        return ""
    return MAILBOX.read_text(encoding="utf-8")

def get_handoff(tail):
    for line in reversed(tail.splitlines()):
        m = HANDOFF_RE.search(line)
        if m:
            return m.group(1).strip()
    return None

def get_state(tail):
    for line in reversed(tail.splitlines()):
        m = STATE_RE.search(line)
        if m:
            return m.group(1).strip()
    return None

def next_msg_b():
    text = read_full()
    max_n = 9
    for m in MSG_NUM_RE.findall(text):
        max_n = max(max_n, int(m))
    return max_n + 1

def append_mailbox(body):
    with open(MAILBOX, "a", encoding="utf-8") as f:
        f.write("\n\n" + body)

def check_gpu():
    code, out = run_cmd([CUDA_PYTHON, "-c",
        "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.version.cuda}|{torch.cuda.device_count()}')"],
        timeout=10)
    if code == 0:
        parts = out.strip().split("|")
        return f"torch={parts[0]} cuda={parts[1]} cu={parts[2]} n_gpu={parts[3]}"
    return f"GPU check failed: {out[:100]}"

def run_training(curriculum, steps=300000, suffix=""):
    label = f"curri{curriculum}" if curriculum is not None else "default"
    run_name = f"VIC_PPO_{label}_{steps//1000}k{suffix}"
    
    args = [
        CUDA_PYTHON, str(TRAIN_SCRIPT),
        "--algo", "PPO",
        "--steps", str(steps),
        "--n-envs", "4",
        "--run-name", run_name,
        "--device", "cuda",
        "--test-damping", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0",
        "--eval-episodes", "20",
    ]
    if curriculum is not None and curriculum < 999:
        args += ["--curriculum-level", str(curriculum)]
    
    log(f"TRAIN start: {run_name}")
    t0 = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=7200, cwd=str(TRAIN_SCRIPT_DIR))
        code, output = r.returncode, (r.stdout[-2000:] + r.stderr[-1000:])
    except subprocess.TimeoutExpired:
        code, output = -1, "[timeout]"
    except Exception as e:
        code, output = -1, str(e)
    dt = time.time() - t0
    
    gen_path = RUNS_DIR / run_name / "generalization_results.json"
    if gen_path.exists():
        try:
            results = json.loads(gen_path.read_text())
            sr_list = [f"d={v['damping']}:{v['success_rate']:.0%}" for k,v in sorted(results.items())]
            k_list = [f"d={v['damping']}:{v['mean_K']:.0f}" for k,v in sorted(results.items())]
            summary = f"{run_name}: {dt/60:.0f}min, SR={sr_list}, K={k_list}"
        except:
            summary = f"{run_name}: {dt/60:.0f}min, results parse failed"
    else:
        summary = f"{run_name}: {dt/60:.0f}min, returncode={code}"
    
    log(summary)
    return code == 0, summary, run_name


def work_cycle(cycle_num):
    log(f"===== Cycle {cycle_num} =====")
    
    ok, out = git("pull", "origin", "session/routeB-cartesian-vic")
    if not ok:
        log(f"git pull failed: {out}")
        return False
    log(f"git pull ok")
    
    tail = read_tail()
    target = get_handoff(tail)
    state = get_state(tail)
    log(f"state=[{state}] handoff->{target}")
    
    if target and ("B" in target or "显卡机" in target or "看门狗" in target):
        log(f"B's turn! Working...")
        
        gpu_info = check_gpu()
        log(f"GPU: {gpu_info}")
        
        # Check if there are remaining curriculum phases
        # Look for completed phases in mailbox
        full = read_full()
        phases_done = set()
        # Match "Phase 1", "Phase 2", "Phase 1+2/4" etc
        for m in re.finditer(r"Phase\s*((?:\d+\+)*\d+)\s*/\s*4", full):
            parts = m.group(1).split("+")
            for p in parts:
                if p.strip().isdigit():
                    phases_done.add(int(p.strip()))
        # Also match "Phase X done" patterns
        for m in re.finditer(r"(?:Phase|P)\s*(\d+)\s*(?:done|complete|\u2713)", full):
            phases_done.add(int(m.group(1)))
        
        log(f"Phases done: {phases_done}")
        
        phase_configs = [
            (1, 0, 300000),
            (2, 1, 150000),
            (3, 2, 100000),
            (4, 999, 50000),  # 999 = None (no curriculum)
        ]
        
        results = []
        for phase_num, curriculum, steps in phase_configs:
            if phase_num in phases_done:
                log(f"Phase {phase_num} already done, skip")
                continue
            log(f"Starting Phase {phase_num} (curriculum={'None' if curriculum==999 else curriculum})")
            ok, summary, rn = run_training(curriculum, steps)
            results.append((phase_num, summary, rn))
            phases_done.add(phase_num)
        
        if results:
            msg_n = next_msg_b()
            report = f"## MSG-B{msg_n} - Watchdog: phases completed\n\nGPU: {gpu_info}\n\n| Phase | Result |\n|:----|:-------|\n"
            for pn, s, rn in results:
                status = "+" if "SR=" in s else "?"
                report += f"| P{pn} {status} | {s} |\n"
            
            report += "\n【状态】[running] (watchdog continuing)\n"
            # Check if all phases done
            if len(phases_done) >= 4:
                report += "【交接棒】-> Please wake AI (SoniXChat) for review\n"
            else:
                report += "【交接棒】-> Watchdog auto-continues\n"
            
            append_mailbox(report)
            git("add", "AI_CHANNEL.md")
            git("commit", "-m", f"MSG-B{msg_n}: watchdog phase results")
            git("push", "origin", "session/routeB-cartesian-vic")
            log(f"Results pushed")
        
        return True
    else:
        log(f"Not B's turn (target={target})")
        return False


def main():
    log("[DOG] RouteB watchdog started (24h loop, hourly)")
    log(f"GPU: {check_gpu()}")
    log(f"Python: {CUDA_PYTHON}")
    
    cycle = 0
    start = time.time()
    max_dur = 24 * 3600
    
    while time.time() - start < max_dur:
        cycle += 1
        try:
            work_cycle(cycle)
        except Exception as e:
            log(f"Cycle {cycle} error: {e}")
        
        elapsed = time.time() - start
        rem = max_dur - elapsed
        log(f"Ran {elapsed/3600:.1f}h, remaining {rem/3600:.1f}h")
        log(f"Sleep 60min...\n")
        
        for _ in range(60):
            time.sleep(60)
            if time.time() - start >= max_dur:
                break
    
    log("[DOG] 24h done, exiting")


if __name__ == "__main__":
    main()
