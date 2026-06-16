#!/usr/bin/env python3
"""Submit the benchmark matrix to SLURM (kiz0) — or run it locally.

Reads ``configs/matrix.yaml`` + ``slurm/kiz0.env``, then for each
``(model x dataset)`` renders the sbatch template and submits one job. A final
aggregation job (dependency ``afterok`` on all benchmark jobs) renders the
color-coded leaderboard (HTML + Markdown).

Modes:
    (default)   submit one SLURM job per (model, dataset)
    --group-by model   one SLURM job per model (all datasets sequentially)
    --local            run sequentially on this machine (no SLURM)
    --dry-run          render + print sbatch scripts, submit nothing
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv  # noqa: E402

from src.benchmark.config import Job, iter_jobs, load_config  # noqa: E402

_SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
_EMPTY_SBATCH_RE = re.compile(r"^#SBATCH --[\w-]+=\s*$")


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple ``KEY="value"`` env file (ignores comments/blank lines)."""
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def render_script(
    template: str,
    env: Dict[str, str],
    job_name: str,
    commands: str,
    *,
    gres: Optional[str] = None,
    time_limit: Optional[str] = None,
) -> str:
    """Fill the sbatch template; drop #SBATCH lines whose value is empty."""
    values = {
        "JOB_NAME": job_name,
        "PARTITION": env.get("PARTITION", ""),
        "GRES": gres if gres is not None else env.get("GRES", ""),
        "QOS": env.get("QOS", ""),
        "ACCOUNT": env.get("ACCOUNT", ""),
        "TIME_LIMIT": time_limit or env.get("TIME_LIMIT", "04:00:00"),
        "LOG_DIR": env.get("LOG_DIR", "slurm/logs"),
        "PROJECT_DIR": str(PROJECT_DIR),
        "MODULE_LOAD": env.get("MODULE_LOAD", ""),
        "VENV_PATH": env.get("VENV_PATH", ".venv"),
        "COMMANDS": commands,
    }
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    # Drop optional #SBATCH lines (e.g. --qos=, --account=, --gres=) left empty.
    kept = [ln for ln in text.splitlines() if not _EMPTY_SBATCH_RE.match(ln)]
    return "\n".join(kept) + "\n"


def _benchmark_command(job: Job, max_samples: Optional[int]) -> str:
    args = job.to_cli_args(max_samples=max_samples)
    return "python scripts/benchmark.py " + " ".join(shlex.quote(a) for a in args)


def _group_jobs(jobs: List[Job], group_by: str) -> Dict[str, List[Job]]:
    groups: Dict[str, List[Job]] = {}
    for job in jobs:
        key = job.model_tag if group_by == "model" else f"{job.model_tag}__{job.dataset}"
        groups.setdefault(key, []).append(job)
    return groups


def _sbatch(script_path: Path, dependency: Optional[str] = None) -> Optional[str]:
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script_path))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        print("ERROR: 'sbatch' not found — are you on the cluster? Use --local.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR submitting {script_path.name}: {e.stderr}", file=sys.stderr)
        return None
    m = _SUBMITTED_RE.search(out)
    job_id = m.group(1) if m else None
    print(f"  submitted {script_path.name} -> job {job_id}")
    return job_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the benchmark matrix to SLURM (kiz0) or run it locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/matrix.yaml")
    parser.add_argument("--kiz0-env", type=Path, default=PROJECT_DIR / "slurm/kiz0.env")
    parser.add_argument("--template", type=Path, default=PROJECT_DIR / "slurm/benchmark.sbatch.template")
    parser.add_argument("--jobs-dir", type=Path, default=PROJECT_DIR / "slurm/jobs")
    parser.add_argument("--group-by", choices=["combo", "model"], default="combo")
    parser.add_argument("--local", action="store_true", help="Run sequentially without SLURM.")
    parser.add_argument("--dry-run", action="store_true", help="Render + print scripts, submit nothing.")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip the leaderboard aggregation job.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples (smoke testing).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    config = load_config(args.config)
    jobs = iter_jobs(config)
    groups = _group_jobs(jobs, args.group_by)
    print(f"{len(jobs)} (model x dataset) jobs in {len(groups)} SLURM group(s).")

    if args.local:
        return _run_local(jobs, args.max_samples, args.no_aggregate)

    env = parse_env_file(args.kiz0_env)
    template = args.template.read_text()
    args.jobs_dir.mkdir(parents=True, exist_ok=True)

    job_ids: List[str] = []
    for group_key, group_jobs in groups.items():
        commands = "\n".join(_benchmark_command(j, args.max_samples) for j in group_jobs)
        script = render_script(template, env, job_name=f"asr-{group_key}", commands=commands)
        script_path = args.jobs_dir / f"{group_key}.sbatch"
        script_path.write_text(script)
        if args.dry_run:
            print(f"\n----- {script_path} -----\n{script}")
            continue
        job_id = _sbatch(script_path)
        if job_id:
            job_ids.append(job_id)

    if args.no_aggregate:
        return 0

    agg_commands = "python scripts/leaderboard.py --export html,md"
    agg_script = render_script(
        template, env, job_name="asr-leaderboard", commands=agg_commands,
        gres="", time_limit="00:15:00",  # no GPU, short
    )
    agg_path = args.jobs_dir / "aggregate_leaderboard.sbatch"
    agg_path.write_text(agg_script)
    if args.dry_run:
        print(f"\n----- {agg_path} (depends on all jobs) -----\n{agg_script}")
        return 0
    if job_ids:
        _sbatch(agg_path, dependency=":".join(job_ids))
    else:
        print("No benchmark jobs submitted; skipping aggregation.", file=sys.stderr)
    return 0


def _run_local(jobs: List[Job], max_samples: Optional[int], no_aggregate: bool) -> int:
    for i, job in enumerate(jobs, start=1):
        print(f"\n=== [{i}/{len(jobs)}] {job.model} on {job.dataset} ===")
        cmd = ["python", "scripts/benchmark.py"] + job.to_cli_args(max_samples=max_samples)
        rc = subprocess.run(cmd, cwd=PROJECT_DIR).returncode
        if rc != 0:
            print(f"  job failed (rc={rc}); continuing", file=sys.stderr)
    if not no_aggregate:
        subprocess.run(
            ["python", "scripts/leaderboard.py", "--export", "html,md"], cwd=PROJECT_DIR
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
