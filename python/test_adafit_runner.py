#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="adafit_test_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "pts.xyz"
        output_path = tmp_path / "normals.xyz"
        input_path.write_text("0 0 0\n1 0 0\n0 1 0\n", encoding="ascii")

        cmd = [
            "python3",
            "python/adafit_runner.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            raise SystemExit("Runner should fail in strict mode when --repo is missing")
        print("strict mode error path OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


