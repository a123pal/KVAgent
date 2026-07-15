from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sweep"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for pad_bytes in (16_384, 65_536, 262_144):
        for fanout in (2, 3, 4):
            output = args.output_dir / f"bytes-{pad_bytes}_fanout-{fanout}.json"
            command = [
                "kvagent-bench",
                "--artifact",
                str(args.artifact),
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                "--pad-bytes",
                str(pad_bytes),
                "--fanout",
                str(fanout),
                "--repetitions",
                "3",
                "--max-tokens",
                "16",
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True)
            payload = json.loads(output.read_text())
            cases.append(payload)

    (args.output_dir / "all.json").write_text(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
