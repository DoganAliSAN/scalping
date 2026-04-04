from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUNNER_PID_PATH = BASE_DIR / "paper_live_runner.pid"
RUNNER_LOG_PATH = BASE_DIR / "paper_live_runner.log"
RUNNER_SCRIPT_PATH = BASE_DIR / "paper_live_logger.py"
RUNNER_PATTERN = "paper_live_logger.py --loop"


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid_payload() -> dict[str, Any] | None:
    if not RUNNER_PID_PATH.exists():
        return None
    try:
        payload = json.loads(RUNNER_PID_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_pid_payload(payload: dict[str, Any]) -> None:
    RUNNER_PID_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _remove_pid_file() -> None:
    if RUNNER_PID_PATH.exists():
        RUNNER_PID_PATH.unlink()


def _discover_loop_processes() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", RUNNER_PATTERN],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1] if len(parts) > 1 else ""
        if pid == os.getpid():
            continue
        rows.append({"pid": pid, "command": command})
    return rows


def _parse_execution_mode(command: str) -> str:
    if "--execution-mode binance" in command:
        return "binance"
    if "--execution-mode paper" in command:
        return "paper"
    return "unknown"


def get_live_runner_status() -> dict[str, Any]:
    payload = _read_pid_payload()
    managed_pid = int(payload.get("pid", 0)) if payload and str(payload.get("pid", "")).isdigit() else None
    managed_running = bool(managed_pid and _pid_is_running(managed_pid))

    if payload and not managed_running:
        _remove_pid_file()
        payload = None
        managed_pid = None

    discovered = _discover_loop_processes()
    managed = None
    if payload and managed_running:
        managed = {
            "pid": managed_pid,
            "command": str(payload.get("command", "")),
            "execution_mode": str(payload.get("execution_mode", "unknown")),
            "started_at_utc": str(payload.get("started_at_utc", "")),
            "log_path": str(payload.get("log_path", str(RUNNER_LOG_PATH))),
        }

    unmanaged = [row for row in discovered if row.get("pid") != managed_pid]
    primary = managed or (unmanaged[0] if unmanaged else None)

    return {
        "running": bool(primary),
        "managed": bool(managed),
        "pid": int(primary["pid"]) if primary else None,
        "execution_mode": str(primary.get("execution_mode") or _parse_execution_mode(str(primary.get("command", "")))) if primary else "stopped",
        "command": str(primary.get("command", "")) if primary else "",
        "started_at_utc": str(primary.get("started_at_utc", "")) if primary else "",
        "log_path": str(primary.get("log_path", str(RUNNER_LOG_PATH))) if primary else str(RUNNER_LOG_PATH),
        "discovered_processes": discovered,
    }


def _terminate_pid(pid: int, *, force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pid, sig)
    except Exception:
        try:
            os.kill(pid, sig)
        except OSError:
            return


def stop_live_runner(*, force: bool = False) -> dict[str, Any]:
    status = get_live_runner_status()
    targets = [row["pid"] for row in status.get("discovered_processes", [])]
    if not targets and status.get("pid"):
        targets = [int(status["pid"])]

    if not targets:
        _remove_pid_file()
        return {"stopped": False, "message": "No live runner process found."}

    for pid in targets:
        _terminate_pid(int(pid), force=False)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not any(_pid_is_running(int(pid)) for pid in targets):
            break
        time.sleep(0.2)

    survivors = [int(pid) for pid in targets if _pid_is_running(int(pid))]
    if survivors and force:
        for pid in survivors:
            _terminate_pid(pid, force=True)

    _remove_pid_file()
    return {
        "stopped": True,
        "pids": targets,
        "forced": bool(survivors and force),
        "message": "Live runner stop signal sent.",
    }


def start_live_runner(*, execution_mode: str = "binance", poll_seconds: int = 300) -> dict[str, Any]:
    current = get_live_runner_status()
    if current.get("running"):
        stop_live_runner(force=True)

    RUNNER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = RUNNER_LOG_PATH.open("a", encoding="utf-8")
    command = [
        sys.executable,
        str(RUNNER_SCRIPT_PATH),
        "--loop",
        "--execution-mode",
        execution_mode,
        "--poll-seconds",
        str(int(poll_seconds)),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )

    payload = {
        "pid": process.pid,
        "execution_mode": execution_mode,
        "command": " ".join(command),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(RUNNER_LOG_PATH),
    }
    _write_pid_payload(payload)
    return {
        "started": True,
        **payload,
    }