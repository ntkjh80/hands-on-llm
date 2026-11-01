#!/usr/bin/env python3
"""Run the twelve CPU concept labs, preserving their actual output."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, choices=range(1, 13), help="Run one chapter only")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per lab in seconds")
    parser.add_argument("--save", action="store_true", help="Replace lab-output.txt with this run's output")
    args = parser.parse_args()
    chapters = [args.chapter] if args.chapter else range(1, 13)
    failed = []
    for number in chapters:
        folder = ROOT / "chapters" / f"ch{number:02d}"
        script = folder / "lab.py"
        if not script.exists():
            failed.append(number)
            print(f"ch{number:02d}: MISSING lab.py")
            continue
        try:
            run = subprocess.run([sys.executable, str(script)], cwd=folder,
                                 capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            failed.append(number)
            print(f"ch{number:02d}: TIMEOUT ({args.timeout}s)")
            continue
        if args.save:
            (folder / "lab-output.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
        print(f"ch{number:02d}: {'PASS' if run.returncode == 0 else 'FAIL'} "
              f"({len(run.stdout.splitlines())} output lines)")
        if run.returncode:
            failed.append(number)
            print(run.stderr[-3000:])
    if failed:
        print("Failed:", ", ".join(f"ch{x:02d}" for x in failed))
        return 1
    print("All requested labs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
