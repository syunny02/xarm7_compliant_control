#!/usr/bin/env python3
"""
Route B Smart Watchdog - автономный агент с навыками
=====================================================
Каждый час проверяет почтовый ящик. Когда очередь B:
1. Читает задание от AI
2. Выбирает навык: train / adjust_env / analyze / eval
3. Выполняет
4. Пишет отчёт и push

Может:
- Запускать GPU обучение (любые параметры)
- Менять константы в cartesian_vic_env.py (anti-deg, reward и т.д.)
- Читать и сравнивать результаты
- Автоматически итеративно улучшать параметры
"""

import os, sys, subprocess, re, time, json, shutil
from pathlib import Path
from datetime import datetime

# ── Пути ──
REPO_DIR = Path(__file__).parent.resolve()
MAIN_REPO = REPO_DIR.parent
MAILBOX = REPO_DIR / "AI_CHANNEL.md"
ENV_FILE = MAIN_REPO / "xarm7_cartesian_vic" / "cartesian_vic_env.py"
TRAIN_SCRIPT = MAIN_REPO / "xarm7_cartesian_vic" / "train_vic.py"
TRAIN_CWD = MAIN_REPO / "xarm7_cartesian_vic"
RUNS_DIR = TRAIN_CWD / "runs"
CUDA_PYTHON = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"

# ── Регулярки для信箱 ──
HANDOFF_RE = re.compile(r"\u3010\u4ea4\u63a5\u68d2\u3011\u2192\s*(.+?)(?:\n|$)")
MSG_NUM_RE = re.compile(r"MSG-B(\d+)")

# ══════════════════════════════════════════════════
#  НАВЫК 1: Чтение почтового ящика
# ══════════════════════════════════════════════════

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def git(*args, timeout=30):
    try:
        r = subprocess.run(["git", "-C", str(REPO_DIR)] + list(args),
            capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip()
    except Exception as e:
        return False, str(e)

def read_mailbox():
    if not MAILBOX.exists(): return ""
    return MAILBOX.read_text(encoding="utf-8")

def get_handoff(text):
    for line in reversed(text.splitlines()):
        m = HANDOFF_RE.search(line)
        if m: return m.group(1).strip()
    return None

def next_msg_b(text):
    max_n = 11
    for m in MSG_NUM_RE.findall(text):
        max_n = max(max_n, int(m))
    return max_n + 1

def append_mailbox(body):
    with open(MAILBOX, "a", encoding="utf-8") as f:
        f.write("\n\n" + body)

# ══════════════════════════════════════════════════
#  НАВЫК 2: Изменение констант в env файле
# ══════════════════════════════════════════════════

def set_env_constant(name, value):
    """Изменить константу в cartesian_vic_env.py. value может быть числом или строкой."""
    if not ENV_FILE.exists():
        log(f"ENV_FILE not found: {ENV_FILE}")
        return False
    
    content = ENV_FILE.read_text(encoding="utf-8")
    
    # Пробуем разные форматы: NAME = value, NAME=value
    patterns = [
        rf"^{name}\s*=\s*[0-9.-]+" ,          # FLOAT = 0.001
        rf"^{name}\s*=\s*[0-9]+"   ,           # INT = 1000
        rf"{name}\s*=\s*[0-9.-]+"  ,           # с отступом
        rf"{name}\s*=\s*[0-9]+"    ,
    ]
    
    for pat in patterns:
        new_val = str(value)
        if isinstance(value, float):
            # форматируем: 0.1 или 1.0
            new_val = f"{value}" if '.' in str(value) else f"{value}.0"
        
        new_content = re.sub(pat, f"{name} = {new_val}", content, count=1)
        if new_content != content:
            ENV_FILE.write_text(new_content, encoding="utf-8")
            log(f"Changed {name} = {new_val}")
            return True
    
    log(f"Could not find {name} in env file")
    return False

def read_env_constant(name):
    """Прочитать текущее значение константы из env файла."""
    if not ENV_FILE.exists():
        return None
    content = ENV_FILE.read_text(encoding="utf-8")
    m = re.search(rf"{name}\s*=\s*([0-9.-]+(?:e[+-]?\d+)?)", content)
    if m:
        val = m.group(1)
        return float(val) if '.' in val or 'e' in val else int(val)
    return None

# ══════════════════════════════════════════════════
#  НАВЫК 3: Запуск обучения
# ══════════════════════════════════════════════════

def run_training(steps=300000, curriculum=None, run_name=None, device="cuda", test_damping=None):
    """Запустить PPO обучение. Возвращает (ok, summary, run_name)."""
    if run_name is None:
        label = f"curri{curriculum}" if curriculum is not None else "default"
        run_name = f"VIC_PPO_{label}_{steps//1000}k"
    
    cmd = [CUDA_PYTHON, str(TRAIN_SCRIPT),
        "--algo", "PPO",
        "--steps", str(steps),
        "--n-envs", "4",
        "--run-name", run_name,
        "--device", device,
    ]
    if curriculum is not None and curriculum < 999:
        cmd += ["--curriculum-level", str(curriculum)]
    if test_damping:
        cmd += ["--test-damping"] + [str(d) for d in test_damping] + ["--eval-episodes", "20"]
    
    log(f"TRAIN {run_name} ({steps}k steps)")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(TRAIN_CWD))
        ok = r.returncode == 0
        output = r.stdout[-2000:] + r.stderr[-1000:]
    except subprocess.TimeoutExpired:
        ok, output = False, "[timeout]"
    except Exception as e:
        ok, output = False, str(e)
    dt = time.time() - t0
    
    gen_path = RUNS_DIR / run_name / "generalization_results.json"
    if gen_path.exists():
        try:
            results = json.loads(gen_path.read_text())
            k_avg = sum(v['mean_K'] for v in results.values()) / len(results)
            sr_best = max(v['success_rate'] for v in results.values())
            door_best = max(v['mean_door_ang'] for v in results.values())
            summary = f"{run_name}: {dt/60:.0f}min K={k_avg:.0f} SR={sr_best:.0%} door={door_best:.3f}"
        except:
            summary = f"{run_name}: {dt/60:.0f}min (parse fail)"
    else:
        summary = f"{run_name}: {dt/60:.0f}min ok={ok}"
    
    log(summary)
    return ok, summary, run_name

# ══════════════════════════════════════════════════
#  НАВЫК 4: Анализ результатов обучения
# ══════════════════════════════════════════════════

def analyze_results(run_name):
    """Прочитать generalization_results.json и вернуть сводку."""
    path = RUNS_DIR / run_name / "generalization_results.json"
    if not path.exists():
        return None
    
    data = json.loads(path.read_text())
    items = sorted(data.items(), key=lambda x: x[1]['damping'])
    
    table = f"| Damping | Door(rad) | SR | K | Force(N) |\n|:------:|:---------:|:--:|:--:|:--------:|\n"
    k_sum = 0
    for k, v in items:
        table += f"| {v['damping']:.1f} | {v['mean_door_ang']:.3f} | {v['success_rate']:.0%} | {v['mean_K']:.0f} | {v['mean_contact_force']:.1f} |\n"
        k_sum += v['mean_K']
    k_avg = k_sum / len(items)
    
    return {
        "table": table,
        "K_avg": k_avg,
        "SR_max": max(v['success_rate'] for v in data.values()),
        "door_max": max(v['mean_door_ang'] for v in data.values()),
        "items": items,
    }

# ══════════════════════════════════════════════════
#  НАВЫК 5: Анти-дегенерация - автонастройка
# ══════════════════════════════════════════════════

def auto_tune_anti_deg(target_K=300):
    """
    Автоматически подбирает LAMBDA_K_HIGH для достижения целевого K.
    Использует бинарный поиск: пробует разные значения, тренирует 50k, смотрит K.
    """
    log(f"Auto-tuning anti-deg for target K={target_K}")
    
    # Читаем текущие параметры
    current_K_high = read_env_constant("LAMBDA_K_HIGH")
    current_K_low = read_env_constant("LAMBDA_K_LOW")
    log(f"Current: LAMBDA_K_HIGH={current_K_high}, LAMBDA_K_LOW={current_K_low}")
    
    # Пробуем: увеличить LAMBDA_K_HIGH (снять gating)
    # Сначала снимаем gating: LAMBDA_K_HIGH теперь unconditional (убираем if in_contact)
    # Но это сложно через regex. Проще: увеличить LAMBDA_K_HIGH в 10/100/1000 раз
    
    candidates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = []
    
    for lam in candidates:
        set_env_constant("LAMBDA_K_HIGH", lam)
        ok, summary, rn = run_training(50000, None, f"tune_K_lam{lam}", device="cuda", test_damping=[1.0])
        analysis = analyze_results(rn)
        if analysis:
            results.append((lam, analysis['K_avg'], analysis['SR_max']))
            log(f"  lam={lam}: K={analysis['K_avg']:.0f}, SR={analysis['SR_max']:.0%}")
    
    # Восстанавливаем оригинал
    if current_K_high:
        set_env_constant("LAMBDA_K_HIGH", current_K_high)
    
    # Находим лучшее
    if results:
        best = min(results, key=lambda x: abs(x[1] - target_K))
        log(f"Best: LAMBDA_K_HIGH={best[0]}, K={best[1]:.0f}, SR={best[2]:.0%}")
        return best[0]
    return None

# ══════════════════════════════════════════════════
#  ОСНОВНОЙ ЦИКЛ
# ══════════════════════════════════════════════════

def parse_task(text):
    """Парсит последнее сообщение от AI, определяет задачу."""
    # Ищем ключевые слова
    tasks = []
    
    if re.search(r"LAMBDA_K|lambda|anti.deg|anti.degeneration|K_high|K_low", text, re.I):
        tasks.append("tune_anti_deg")
    if re.search(r"(?:train|обуча|тренир|запусти)\s+(?:curri|PPO|\d+k)", text, re.I):
        tasks.append("train")
    if re.search(r"(?:анализ|analyze|сравн|compare|eval|оцен)", text, re.I):
        tasks.append("analyze")
    if re.search(r"(?:измен|change|set|установ|постав|прав|fix|исправ)", text, re.I):
        tasks.append("modify")
    
    # Извлекаем числа
    params = {}
    m = re.search(r"LAMBDA_K_HIGH[=:\s]+([0-9.]+)", text)
    if m: params["LAMBDA_K_HIGH"] = float(m.group(1))
    m = re.search(r"LAMBDA_K_LOW[=:\s]+([0-9.]+)", text)
    if m: params["LAMBDA_K_LOW"] = float(m.group(1))
    m = re.search(r"steps[=:\s]+(\d+)k", text)
    if m: params["steps"] = int(m.group(1)) * 1000
    m = re.search(r"curriculum[=:\s]+(\d+)", text)
    if m: params["curriculum"] = int(m.group(1))
    
    return tasks, params


def work_cycle(cycle):
    log(f"===== Cycle {cycle} =====")
    
    ok, out = git("pull", "origin", "session/routeB-cartesian-vic")
    if not ok:
        log(f"git pull failed: {out}")
        return False
    log("git pull ok")
    
    mailbox = read_mailbox()
    target = get_handoff(mailbox)
    log(f"Handoff -> {target}")
    
    # Проверяем, наша ли очередь
    is_b_turn = target and ("B" in target or "显卡机" in target or "看门狗" in target or "GPU" in target)
    
    if not is_b_turn:
        log("Not B's turn, sleeping")
        return False
    
    log("B's turn! Analyzing task...")
    tasks, params = parse_task(mailbox)
    log(f"Tasks detected: {tasks}, params: {params}")
    
    msg_n = next_msg_b(mailbox)
    report_parts = []
    
    # ── Выполняем задачи ──
    if "tune_anti_deg" in tasks and "LAMBDA_K_HIGH" in params:
        # AI указал конкретное значение
        val = params["LAMBDA_K_HIGH"]
        set_env_constant("LAMBDA_K_HIGH", val)
        report_parts.append(f"Set LAMBDA_K_HIGH = {val}")
        
        # Запускаем проверочное обучение
        ok, summary, rn = run_training(50000, None, f"verify_lam{val}", device="cuda", test_damping=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        analysis = analyze_results(rn)
        if analysis:
            report_parts.append(f"Verification: K_avg={analysis['K_avg']:.0f}, SR_max={analysis['SR_max']:.0%}")
            report_parts.append(analysis['table'])
    
    elif "tune_anti_deg" in tasks and "auto" in mailbox:
        # Автоподбор
        best_lam = auto_tune_anti_deg(300)
        if best_lam:
            set_env_constant("LAMBDA_K_HIGH", best_lam)
            report_parts.append(f"Auto-tuned: LAMBDA_K_HIGH = {best_lam}")
    
    elif "train" in tasks:
        steps = params.get("steps", 300000)
        curriculum = params.get("curriculum", None)
        ok, summary, rn = run_training(steps, curriculum, device="cuda",
            test_damping=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        analysis = analyze_results(rn)
        if analysis:
            report_parts.append(summary)
            report_parts.append(analysis['table'])
    
    elif "analyze" in tasks:
        # Ищем последний run
        runs = sorted(RUNS_DIR.glob("VIC_PPO_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            rn = runs[0].name
            analysis = analyze_results(rn)
            if analysis:
                report_parts.append(f"Analysis of {rn}:")
                report_parts.append(analysis['table'])
    
    # ── Формируем отчёт ──
    if report_parts:
        report = f"## MSG-B{msg_n} - Watchdog auto report\n\n"
        for part in report_parts:
            report += part + "\n\n"
        report += "【状态】[completed] (watchdog executed task)\n"
        report += "【交接棒】-> Please wake AI (SoniXChat) for review\n"
        
        append_mailbox(report)
        git("add", "AI_CHANNEL.md")
        git("add", str(ENV_FILE))
        git("commit", "-m", f"MSG-B{msg_n}: watchdog auto-executed {','.join(tasks)}")
        git("push", "origin", "session/routeB-cartesian-vic")
        log("Report pushed")
    
    return True


def main():
    log("[DOG] SMART watchdog started (24h)")
    log(f"GPU: ", end="")
    
    # Проверка GPU
    r = subprocess.run([CUDA_PYTHON, "-c", "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
        capture_output=True, text=True, timeout=10)
    log(f"Device: {r.stdout.strip()}")
    
    cycle = 0
    start = time.time()
    end = start + 24 * 3600
    
    while time.time() < end:
        cycle += 1
        try:
            work_cycle(cycle)
        except Exception as e:
            log(f"Error cycle {cycle}: {e}")
            import traceback
            traceback.print_exc()
        
        remaining = (end - time.time()) / 3600
        log(f"Remaining: {remaining:.1f}h, sleep 60min...\n")
        
        for _ in range(60):
            time.sleep(60)
            if time.time() >= end:
                break
    
    log("[DOG] 24h expired")

if __name__ == "__main__":
    main()
