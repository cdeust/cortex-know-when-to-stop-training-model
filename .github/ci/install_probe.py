"""Install the built wheel in a clean venv and execute its public API.

Source: pyproject.toml hatch wheel target. Dependencies are not locked in this repo.
"""

from pathlib import Path
import subprocess
import tempfile


def run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="abstain-install-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        interpreter = environment / "bin" / "python"
        run("uv", "build", "--wheel", "--out-dir", str(work / "dist"), cwd=root)
        wheels = list((work / "dist").glob("*.whl"))
        assert len(wheels) == 1, wheels  # one project wheel, pyproject.toml
        run("uv", "venv", str(environment), cwd=root)
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            str(wheels[0]),
            cwd=work,
        )
        run(str(interpreter), "-I", str(root / ".github/ci/api_probe.py"), cwd=work)


if __name__ == "__main__":
    main()
