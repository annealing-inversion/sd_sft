#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "auto_train_eval"
FREE_MEM_MIB = 5120


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "monitor.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tasks() -> list[str]:
    return sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir())


def gpu_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    mem = {}
    for line in result.stdout.splitlines():
        idx, used = [part.strip() for part in line.split(",", 1)]
        mem[int(idx)] = int(used)
    return mem


def training_done(style: str) -> bool:
    out = ROOT / "outputs" / "sdxl_lora" / style
    return (
        (out / "pytorch_lora_weights.safetensors").is_file()
        and (out / "learned_embeds.safetensors").is_file()
        and (out / "training_metadata.json").is_file()
    )


def output_exists(style: str) -> bool:
    return (ROOT / "outputs" / "sdxl_lora" / style).exists()


def eval_done(style: str) -> bool:
    return any((ROOT / "outputs").glob(f"eval_{style}_*/base_lora_scale_grid.png")) or any(
        (ROOT / "outputs").glob(f"eval_{style}_*/cardcaptor_sakura_base_lora_scale_grid_v2.png")
    )


def start_process(name: str, gpu: int, cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}_{stamp()}.log"
    exit_path = LOG_DIR / f"{name}_{stamp()}.exit"
    wrapped = ["bash", "-lc", " ".join(subprocess.list2cmdline([part]) for part in cmd) + f"; code=$?; echo $code > {exit_path}; exit $code"]
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        wrapped,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    (LOG_DIR / f"{name}.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    (LOG_DIR / f"{name}.logpath").write_text(str(log_path) + "\n", encoding="utf-8")
    log(f"started {name} on gpu {gpu}: pid={process.pid}, log={log_path}")
    return process


def start_train(style: str, gpu: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "STYLE": style,
            "DATA_DIR": f"data/{style}",
            "OUTPUT_DIR": f"outputs/sdxl_lora/{style}",
            "PLACEHOLDER_TOKEN": f"<{style}_style>",
            "NUM_PROCESSES": "1",
            "MAIN_PROCESS_PORT": str(29510 + gpu),
        }
    )
    return start_process(f"train_{style}", gpu, ["bash", "training_scripts/run_sdxl_pti_lora_single.sh"], env)


def start_eval(style: str, gpu: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return start_process(
        f"eval_{style}",
        gpu,
        [
            ".conda/sdxl-lora/bin/python",
            "tools/eval_lora_grid.py",
            "--style",
            style,
            "--lora-dir",
            f"outputs/sdxl_lora/{style}",
        ],
        env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()

    active: dict[str, tuple[str, int, subprocess.Popen]] = {}
    log(f"monitor start, interval={args.interval}s")
    while True:
        for key, (kind, gpu, proc) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log(f"finished {kind} {key} on gpu {gpu}: exit={code}")
                del active[key]

        assigned = {gpu for _, gpu, proc in active.values() if proc.poll() is None}
        mem = gpu_memory()
        free_gpus = [gpu for gpu, used in sorted(mem.items()) if used < FREE_MEM_MIB and gpu not in assigned]

        active_styles = {key.split(":", 1)[1] for key in active}
        for gpu in list(free_gpus):
            pending_eval = [
                style for style in tasks()
                if training_done(style) and not eval_done(style) and style not in active_styles
            ]
            if pending_eval:
                style = pending_eval[0]
                active[f"eval:{style}"] = ("eval", gpu, start_eval(style, gpu))
                free_gpus.remove(gpu)
                active_styles.add(style)

        for gpu in list(free_gpus):
            pending_train = [
                style for style in tasks()
                if not training_done(style) and not output_exists(style) and style not in active_styles
            ]
            if pending_train:
                style = pending_train[0]
                active[f"train:{style}"] = ("train", gpu, start_train(style, gpu))
                free_gpus.remove(gpu)
                active_styles.add(style)

        blocked = [
            style for style in tasks()
            if not training_done(style) and output_exists(style)
        ]
        if blocked:
            log(f"blocked incomplete output dirs, not overwriting: {', '.join(blocked)}")

        if all(training_done(style) and eval_done(style) for style in tasks()):
            log("all training and eval jobs complete")
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
