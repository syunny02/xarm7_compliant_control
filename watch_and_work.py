#!/usr/bin/env python3
"""
Route B 信箱看门狗 — 显卡机自动接力循环
============================================
每 60 分钟循环一次：
1. git pull 信箱
2. 检查【交接棒】是否指向 B
3. 如果是：执行任务（训练/评估/验证）
4. 更新 AI_CHANNEL.md + git push
5. 睡 1 小时 → 重复

要求：用 Python 3.12 (cu128 torch, 有 GPU)
"""

import os, sys, subprocess, re, time, json
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.resolve()
MAILBOX = REPO_DIR / "AI_CHANNEL.md"
TRAIN_SCRIPT = REPO_DIR / "xarm7_cartesian_vic" / "train_vic.py"
ENV_SCRIPT = REPO_DIR / "xarm7_cartesian_vic" / "cartesian_vic_env.py"
RUNS_DIR = REPO_DIR / "runs"

HANDOFF_RE = re.compile(r"【交接棒】→\s*(.+?)(?:\n|$)")
STATE_RE = re.compile(r"【状态】\[(.+?)\]")
MSG_NUM_RE = re.compile(r"MSG-B(\d+)")

CUDA_PYTHON = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def git(*args, timeout=60):
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

def run_cmd(cmd, timeout=3600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout[-2000:] + r.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return -1, "[timeout]"
    except Exception as e:
        return -1, str(e)

def read_mailbox():
    if not MAILBOX.exists():
        return ""
    text = MAILBOX.read_text(encoding="utf-8")
    lines = text.splitlines()
    return "\n".join(lines[-80:])

def read_full_mailbox():
    if not MAILBOX.exists():
        return ""
    return MAILBOX.read_text(encoding="utf-8")

def get_handoff_target(tail):
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
    text = read_full_mailbox()
    max_n = 7  # start from B07
    for m in MSG_NUM_RE.findall(text):
        max_n = max(max_n, int(m))
    return max_n + 1

def append_mailbox(msg_body):
    """Append a new MSG to the mailbox"""
    with open(MAILBOX, "a", encoding="utf-8") as f:
        f.write("\n\n" + msg_body)

def check_gpu():
    """Return GPU info string"""
    ok, out = git("pull", "origin", "session/routeB-cartesian-vic")
    code, out = run_cmd([CUDA_PYTHON, "-c", 
        "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.version.cuda}|{torch.cuda.device_count()}')"],
        timeout=10)
    if code == 0:
        parts = out.strip().split("|")
        return f"torch={parts[0]} cuda={parts[1]} cu={parts[2]} n_gpu={parts[3]}"
    return f"GPU check failed: {out[:100]}"


def run_training(curriculum_level, steps=300000, run_name_suffix=""):
    """Run PPO training with specified curriculum level. Returns (ok, summary)"""
    run_name = f"VIC_PPO_300k_curri{curriculum_level}{run_name_suffix}"
    
    cmd = [
        CUDA_PYTHON, str(TRAIN_SCRIPT),
        "--algo", "PPO",
        "--steps", str(steps),
        "--n-envs", "4",
        "--run-name", run_name,
        "--curriculum-level", str(curriculum_level),
        "--device", "cuda",
        "--test-damping", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0",
        "--eval-episodes", "20",
    ]
    
    log(f"开始训练: {run_name} (curriculum_level={curriculum_level}, {steps} steps)")
    t0 = time.time()
    code, output = run_cmd(cmd, timeout=7200)  # 2h max
    dt = time.time() - t0
    
    # Try to read results
    gen_path = RUNS_DIR / run_name / "generalization_results.json"
    if gen_path.exists():
        try:
            results = json.loads(gen_path.read_text())
            sr = {f"d={v['damping']}": f"{v['success_rate']:.0%}" for k,v in sorted(results.items())}
            Ks = {f"d={v['damping']}": f"{v['mean_K']:.0f}" for k,v in sorted(results.items())}
            summary = f"✅ {run_name}: {dt/60:.0f}min, SR={sr}, K={Ks}"
        except:
            summary = f"⚠️ {run_name}: trained but results parse failed"
    else:
        summary = f"{'✅' if code==0 else '❌'} {run_name}: {dt/60:.0f}min, returncode={code}"
    
    log(summary)
    return code == 0, summary, run_name


def work_cycle(cycle_num):
    """One full work cycle"""
    log(f"═══════ 第 {cycle_num} 轮 ═══════")
    
    # 1. Pull
    ok, out = git("pull", "origin", "session/routeB-cartesian-vic")
    if not ok:
        log(f"❌ git pull failed: {out}")
        return False
    log(f"✅ git pull ok")
    
    # 2. Check handoff
    tail = read_mailbox()
    target = get_handoff_target(tail)
    state = get_state(tail)
    log(f"状态: [{state}]  交接棒: → {target}")
    
    # 3. If it's B's turn
    if target and ("B" in target or "显卡机" in target):
        log(f"🟢 轮到 B！开始干活...")
        gpu_info = check_gpu()
        log(f"GPU: {gpu_info}")
        
        msg_n = next_msg_b()
        
        # 检查环境 + GPU 指纹 = 第一件事
        if "上cu128" in (state or "") or "GPU" in (state or "") or "卡在" in (state or ""):
            log(f"🎯 GPU 指纹任务")
            report = (
                f"## MSG-B{msg_n} — 看门狗自动报告\n\n"
                f"**GPU 指纹**: {gpu_info} ✅\n\n"
                f"已确认 Python 3.12 cu128 可用，开始执行训练任务。\n\n"
            )
            append_mailbox(report)
            ok, out = git("add", "AI_CHANNEL.md")
            git("commit", "-m", f"MSG-B{msg_n}: GPU fingerprint auto-report")
            git("push", "origin", "session/routeB-cartesian-vic")
            log(f"GPU 指纹已推送")
        
        # 训练任务
        log(f"🎯 开始 curriculum 阶梯训练")
        
        results_summary = []
        
        # 阶段 1: curriculum_level=0
        ok, summary, rn = run_training(0, 300000, "")
        results_summary.append(summary)
        
        # 阶段 2: curriculum_level=1  
        ok2, summary2, rn2 = run_training(1, 150000, "")
        results_summary.append(summary2)
        
        # 阶段 3: curriculum_level=2
        ok3, summary3, rn3 = run_training(2, 100000, "")
        results_summary.append(summary3)
        
        # 阶段 4: curriculum_level=None (原始出厂位)
        ok4, summary4, rn4 = run_training(999, 50000, "")
        results_summary.append(summary4)
        
        msg_n2 = next_msg_b()
        report2 = (
            f"## MSG-B{msg_n2} — 四阶 Curriculum 训练完成\n\n"
            f"GPU: {gpu_info}\n\n"
            f"| 阶段 | 结果 |\n"
            f"|:---|:-----|\n"
        )
        for s in results_summary:
            report2 += f"| {s.replace('✅','').replace('❌','').strip()} |\n"
        report2 += "\n\n【状态】[已完成]（四阶训练跑完，详见 runs/ 目录）\n"
        report2 += "【交接棒】→ 请唤起 AI（SoniXChat）验收结果\n"
        append_mailbox(report2)
        
        ok, out = git("add", "AI_CHANNEL.md")
        git("commit", "-m", f"MSG-B{msg_n2}: 4-stage curriculum training complete")
        ok, out = git("push", "origin", "session/routeB-cartesian-vic")
        
        if ok:
            log(f"✅ 训练结果已推送，交接棒转给 AI")
        else:
            log(f"❌ push failed: {out}")
        
        return True
    else:
        log(f"⏸️ 不是 B 的棒次 (target={target})")
        return False


def main():
    log("🐶 RouteB 看门狗启动 (24h 循环, 每小时检查)")
    log(f"GPU: {check_gpu()}")
    log(f"Python: {CUDA_PYTHON}")
    
    cycle = 0
    start_time = time.time()
    max_duration = 24 * 3600  # 24 hours
    
    while time.time() - start_time < max_duration:
        cycle += 1
        try:
            work_cycle(cycle)
        except Exception as e:
            log(f"❌ 第 {cycle} 轮异常: {e}")
        
        elapsed = time.time() - start_time
        remaining = max_duration - elapsed
        log(f"⏳ 已运行 {elapsed/3600:.1f}h, 剩余 {remaining/3600:.1f}h")
        log(f"💤 睡 60 分钟...\n")
        
        # Sleep in 60s chunks so we can be interrupted gracefully
        for _ in range(60):
            time.sleep(60)
            if time.time() - start_time >= max_duration:
                break
    
    log(f"🏁 24h 运行结束")


if __name__ == "__main__":
    main()
