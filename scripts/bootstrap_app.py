#!/usr/bin/env python3
"""End-to-end bootstrap for local app readiness (Steps 1->5)."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/events_processed.parquet"
DEFAULT_INDEX = PROJECT_ROOT / "artifacts/faiss_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all required scripts in order to make the app operational.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Config file for dataset build.",
    )
    parser.add_argument(
        "--index-config",
        default=str(PROJECT_ROOT / "configs/indexing.yaml"),
        help="Config file for index build.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET),
        help="Processed dataset path (parquet) used by index build.",
    )
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX),
        help="FAISS index directory.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--log-level", default="INFO", help="API log level")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip dataset fetch/build and use existing processed dataset.",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="Force dataset rebuild even if file already exists.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip scripts/check_env.py",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        help="Skip scripts/build_index.py",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run preparation steps but do not start API.",
    )
    parser.add_argument(
        "--skip-api-smoke",
        action="store_true",
        help="Skip scripts/api_test.py once API is up.",
    )
    parser.add_argument(
        "--exit-after-smoke",
        action="store_true",
        help="Start API, run smoke test, then stop API and exit.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=45,
        help="Seconds to wait for API /health readiness.",
    )
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m{rem:05.2f}s"


def _run_step(
    *,
    name: str,
    command: Sequence[str],
    durations: list[tuple[str, float]],
    cwd: Path,
) -> None:
    print(f"\n=== {name} ===")
    print("$ " + " ".join(command))
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd)
    elapsed = time.perf_counter() - started
    durations.append((name, elapsed))

    if completed.returncode != 0:
        raise RuntimeError(
            f"Step failed ({name}) with exit code {completed.returncode}."
        )

    print(f"[OK] {name} termine en {_format_duration(elapsed)}")


def _wait_for_health(base_url: str, timeout_seconds: int) -> float:
    health_url = f"{base_url.rstrip('/')}/health"
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last_error = ""

    while time.perf_counter() < deadline:
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return time.perf_counter() - started
            last_error = f"status={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)

    raise TimeoutError(
        f"API did not become healthy within {timeout_seconds}s ({last_error})."
    )


def _print_summary(durations: list[tuple[str, float]]) -> None:
    print("\n=== Resume des durees ===")
    total = 0.0
    for name, elapsed in durations:
        total += elapsed
        print(f"- {name:<28} {_format_duration(elapsed)}")
    print(f"- {'TOTAL':<28} {_format_duration(total)}")


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _dataset_has_minimum_content(dataset_path: Path) -> tuple[bool, str]:
    if not dataset_path.exists():
        return False, "missing file"

    try:
        suffix = dataset_path.suffix.lower()
        if suffix == ".parquet":
            dataframe = pd.read_parquet(dataset_path)
        elif suffix == ".jsonl":
            dataframe = pd.read_json(dataset_path, lines=True)
        elif suffix == ".csv":
            dataframe = pd.read_csv(dataset_path)
        else:
            return False, f"unsupported dataset format: {suffix}"
    except Exception as exc:
        return False, f"failed to read dataset: {exc}"

    if dataframe.empty:
        return False, "dataset is empty"
    if "document_text" not in dataframe.columns:
        return False, "missing required column: document_text"

    non_empty_docs = (
        dataframe["document_text"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    if int(non_empty_docs) <= 0:
        return False, "all document_text values are empty"

    return True, ""


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    python_bin = sys.executable
    durations: list[tuple[str, float]] = []

    dataset_path = Path(args.dataset_path).resolve()
    index_path = Path(args.index_path).resolve()

    try:
        if not args.skip_env_check:
            _run_step(
                name="check_env",
                command=[python_bin, str(PROJECT_ROOT / "scripts/check_env.py")],
                durations=durations,
                cwd=PROJECT_ROOT,
            )

        should_build_dataset = False
        if not args.offline:
            dataset_ok, dataset_reason = _dataset_has_minimum_content(dataset_path)
            should_build_dataset = args.force_dataset or (not dataset_ok)
            if should_build_dataset:
                if dataset_path.exists() and not args.force_dataset:
                    print(
                        "\n[WARN] Dataset present but not usable, rebuild required: "
                        f"{dataset_reason} ({dataset_path})"
                    )
                _run_step(
                    name="build_dataset",
                    command=[
                        python_bin,
                        str(PROJECT_ROOT / "scripts/build_dataset.py"),
                        "--config",
                        str(Path(args.config).resolve()),
                        "--processed-output",
                        str(dataset_path),
                    ],
                    durations=durations,
                    cwd=PROJECT_ROOT,
                )
            else:
                print(
                    "\n[INFO] Dataset deja present, build_dataset saute "
                    f"({dataset_path})."
                )
        else:
            print("\n[INFO] Mode offline actif: build_dataset saute.")

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Processed dataset introuvable. "
                f"Attendu: {dataset_path}. "
                "Lance build_dataset (ou retire --offline)."
            )
        dataset_ok, dataset_reason = _dataset_has_minimum_content(dataset_path)
        if not dataset_ok:
            raise RuntimeError(
                "Processed dataset is not usable for index build "
                f"({dataset_reason}) at {dataset_path}. "
                "Run bootstrap without --offline or use --force-dataset."
            )

        if not args.skip_index_build:
            _run_step(
                name="build_index",
                command=[
                    python_bin,
                    str(PROJECT_ROOT / "scripts/build_index.py"),
                    "--config",
                    str(Path(args.index_config).resolve()),
                    "--input",
                    str(dataset_path),
                    "--output",
                    str(index_path),
                ],
                durations=durations,
                cwd=PROJECT_ROOT,
            )
        else:
            print("\n[INFO] build_index saute (--skip-index-build).")

        if args.prepare_only:
            _print_summary(durations)
            print("\nPreparation terminee. API non demarree (--prepare-only).")
            return 0

        run_api_cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts/run_api.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--log-level",
            args.log_level,
        ]

        print("\n=== run_api ===")
        print("$ " + " ".join(run_api_cmd))
        api_started = time.perf_counter()
        api_process = subprocess.Popen(run_api_cmd, cwd=PROJECT_ROOT)

        try:
            startup_elapsed = _wait_for_health(
                base_url=f"http://{args.host}:{args.port}",
                timeout_seconds=args.startup_timeout,
            )
        except Exception:
            _terminate_process(api_process)
            raise

        durations.append(("run_api_startup", startup_elapsed))
        print(f"[OK] API healthy en {_format_duration(startup_elapsed)}")

        if not args.skip_api_smoke:
            smoke_cmd = [
                python_bin,
                str(PROJECT_ROOT / "scripts/api_test.py"),
                "--base-url",
                f"http://{args.host}:{args.port}",
            ]
            if not os.getenv("MISTRAL_API_KEY", "").strip():
                smoke_cmd.append("--offline")

            admin_token = os.getenv("ADMIN_TOKEN", "").strip()
            if admin_token:
                smoke_cmd.extend(["--admin-token", admin_token])

            _run_step(
                name="api_smoke",
                command=smoke_cmd,
                durations=durations,
                cwd=PROJECT_ROOT,
            )
        else:
            print("\n[INFO] api_test saute (--skip-api-smoke).")

        _print_summary(durations)

        if args.exit_after_smoke:
            _terminate_process(api_process)
            print("\nAPI arretee (--exit-after-smoke).")
            return 0

        print("\nApp operationnelle. API en cours d'execution.")
        print("Arret: Ctrl+C")

        try:
            while True:
                return_code = api_process.poll()
                if return_code is not None:
                    print(f"\nAPI terminee avec code {return_code}.")
                    return return_code
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nArret demande, fermeture de l'API...")
            _terminate_process(api_process)
            return 0

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        _print_summary(durations)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
