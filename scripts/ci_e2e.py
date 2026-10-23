"""Build a wheel and exercise its public API outside the source checkout.

Uses the project's hatch wheel target and declared runtime dependencies.
This covers the explicit heuristic fallback, not trained ONNX inference.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="abstain-e2e-") as directory:
        work = Path(directory)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(work),
                str(root),
            ],
            cwd=work,
            check=True,
        )
        wheels = list(work.glob("*.whl"))
        assert len(wheels) == 1, wheels
        environment = work / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", str(wheels[0])], cwd=work, check=True)
        subprocess.run(
            [str(python), "-I", str(root / "scripts/ci_api_probe.py")],
            cwd=work,
            check=True,
        )


if __name__ == "__main__":
    main()
