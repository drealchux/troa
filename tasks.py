"""
Project task runner: short names for TROA's common commands.

A cross-platform stand-in for a Makefile. Needs only Python, and runs every
command with the same interpreter that runs this file, so the active venv is used.

    python tasks.py                 # list tasks
    python tasks.py test
    python tasks.py eval --no-judge # extra arguments go to the task's last command

Run from anywhere; commands always execute from the repository root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
QDRANT = ["--qdrant-path", "qdrant_local"]
HARNESS = [PY, "-m", "src.eval.harness", *QDRANT]

# name -> (description, list of commands). Extra CLI arguments are appended
# to the last command only.
TASKS: dict[str, tuple[str, list[list[str]]]] = {
    "setup": ("Install dependencies, including test tools, into the current environment",
              [[PY, "-m", "pip", "install", "-r", "requirements-dev.txt"]]),
    "test": ("Run the test suite",
             [[PY, "-m", "pytest", "tests/", "-q"]]),
    "download": ("Download the RRC manuals into data/raw/manual",
                 [[PY, "data/download_data.py"]]),
    "ingest-check": ("Parse and chunk the manuals without embedding (fast sanity check)",
                     [[PY, "-m", "src.ingest.pipeline", "--corpus", "data/raw/manual", "--dry-run"]]),
    "ingest": ("Parse, chunk, embed and store the manuals in qdrant_local/",
               [[PY, "-m", "src.ingest.pipeline", "--corpus", "data/raw/manual", *QDRANT]]),
    "ask": ("Ask a question in the terminal: python tasks.py ask \"your question\"",
            [[PY, "ask.py"]]),
    "dashboard": ("Start the Streamlit dashboard",
                  [[PY, "-m", "streamlit", "run", "dashboard/app.py"]]),
    "api": ("Start the HTTP API on port 8000 (settings from .env)",
            [[PY, "-m", "uvicorn", "src.api.app:app", "--port", "8000"]]),
    "eval-check": ("Validate the eval set YAML without calling the pipeline",
                   [[*HARNESS, "--dry-run"]]),
    "eval": ("Run the eval harness with the Opus judge (baseline settings)",
             [[*HARNESS, "--output", "eval_data/results_latest.jsonl"]]),
    "eval-ablation": ("Run vector, vector + reranker, hybrid, and hybrid + agent loop for comparison",
                      [[*HARNESS, "--output", "eval_data/results_vector.jsonl"],
                       [*HARNESS, "--output", "eval_data/results_vector_rerank.jsonl", "--rerank"],
                       [*HARNESS, "--output", "eval_data/results_hybrid.jsonl",
                        "--search-mode", "hybrid"],
                       [*HARNESS, "--output", "eval_data/results_hybrid_agentic.jsonl",
                        "--search-mode", "hybrid", "--agentic"]]),
    "calibrate": ("Fit the Platt calibrator on judged results into calibration/v1.json",
                  [[PY, "-m", "src.eval.calibration", "train",
                    "--results", "eval_data/results_latest.jsonl",
                    "--output", "calibration/v1.json"]]),
    "docker-up": ("Start Qdrant, Redis and the API with Docker Compose",
                  [["docker", "compose", "up", "-d"]]),
    "docker-down": ("Stop the Docker Compose stack",
                    [["docker", "compose", "down"]]),
}


def list_tasks() -> None:
    print("Usage: python tasks.py <task> [extra args]\n\nTasks:")
    width = max(map(len, TASKS))
    for name, (description, _) in TASKS.items():
        print(f"  {name:<{width}}  {description}")


def run(name: str, extra: list[str]) -> int:
    _, commands = TASKS[name]
    if name == "calibrate":
        (ROOT / "calibration").mkdir(exist_ok=True)
    for i, cmd in enumerate(commands):
        if i == len(commands) - 1:
            cmd = cmd + extra
        print(f"\n> {' '.join(cmd)}", flush=True)
        try:
            code = subprocess.call(cmd, cwd=ROOT)
        except FileNotFoundError:
            print(f"Command not found: {cmd[0]}")
            return 127
        except KeyboardInterrupt:
            return 130
        if code != 0:
            return code
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        list_tasks()
        return 0
    name, extra = sys.argv[1], sys.argv[2:]
    if name not in TASKS:
        print(f"Unknown task: {name}\n")
        list_tasks()
        return 2
    return run(name, extra)


if __name__ == "__main__":
    sys.exit(main())
