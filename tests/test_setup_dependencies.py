"""Verify mandatory and optional dependency setup behavior."""
import os
from pathlib import Path
import shutil
import subprocess

from support import REPO_ROOT, run, workdir


OPTIONAL_REQUIREMENTS = "requirements-aec.txt"
AEC_REQUIREMENT = "pywebrtc-audio>=0.1.0,<0.2"


def _requirement_lines(path):
    return [line.split("#", 1)[0].strip()
            for line in Path(path).read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def test_aec_dependency_is_optional():
    """Putting native AEC in the mandatory set can prevent fallback startup."""
    mandatory = _requirement_lines(REPO_ROOT / "requirements.txt")
    assert not any(line.startswith("pywebrtc-audio") for line in mandatory), \
        "pywebrtc-audio is still mandatory"

    optional_path = REPO_ROOT / OPTIONAL_REQUIREMENTS
    assert optional_path.exists(), f"{OPTIONAL_REQUIREMENTS} is missing"
    assert _requirement_lines(optional_path) == [AEC_REQUIREMENT]


def test_setup_retries_changed_optional_dependencies_without_blocking_launch():
    """An optional native build failure must warn and launch; mandatory failure must stop."""
    with workdir("optional-dependency-setup") as d:
        shutil.copy(REPO_ROOT / "run.sh", d / "run.sh")
        shutil.copy(REPO_ROOT / "requirements.txt", d / "requirements.txt")
        optional = d / OPTIONAL_REQUIREMENTS
        optional.write_text(AEC_REQUIREMENT + "\n")

        fake_python = d / ".venv" / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text("""#!/usr/bin/env bash
if [ "${1:-}" = "-" ]; then
  echo "3.14.0"
  exit 0
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then
  case " $* " in
    *" install --upgrade pip "*) echo upgrade >> "$SETUP_LOG" ;;
    *" install -r requirements.txt "*)
      echo mandatory >> "$SETUP_LOG"
      [ "${FAIL_MANDATORY:-0}" = "1" ] && exit 31
      ;;
    *" install -r requirements-aec.txt "*)
      echo optional >> "$SETUP_LOG"
      exit 32
      ;;
  esac
  exit 0
fi
if [ "${1:-}" = "audiocript.py" ]; then
  echo launched >> "$SETUP_LOG"
  exit 0
fi
exit 99
""")
        fake_python.chmod(0o755)
        log = d / "setup.log"
        env = os.environ.copy()
        env.update({"PYTHON": str(fake_python), "SKIP_DEP_CHECK": "1",
                    "SETUP_LOG": str(log)})

        first = subprocess.run(["bash", str(d / "run.sh")], cwd=d, env=env,
                               capture_output=True, text=True)
        assert first.returncode == 0, first.stdout + first.stderr
        assert log.read_text().splitlines() == [
            "upgrade", "mandatory", "optional", "launched"]
        warning = first.stdout + first.stderr
        assert "Could not install optional WebRTC echo cancellation" in warning
        assert "pip install -r requirements-aec.txt" in warning

        log.write_text("")
        unchanged = subprocess.run(["bash", str(d / "run.sh")], cwd=d, env=env,
                                   capture_output=True, text=True)
        assert unchanged.returncode == 0
        assert log.read_text().splitlines() == ["launched"]

        optional.write_text(AEC_REQUIREMENT + "\n# retry changed optional set\n")
        log.write_text("")
        changed = subprocess.run(["bash", str(d / "run.sh")], cwd=d, env=env,
                                 capture_output=True, text=True)
        assert changed.returncode == 0, changed.stdout + changed.stderr
        assert log.read_text().splitlines() == [
            "upgrade", "mandatory", "optional", "launched"]

        (d / ".venv" / ".requirements.sha").unlink()
        log.write_text("")
        strict_env = env | {"FAIL_MANDATORY": "1"}
        mandatory_failure = subprocess.run(
            ["bash", str(d / "run.sh")], cwd=d, env=strict_env,
            capture_output=True, text=True)
        assert mandatory_failure.returncode == 31
        assert log.read_text().splitlines() == ["upgrade", "mandatory"]


if __name__ == "__main__":
    run(["test_aec_dependency_is_optional",
         "test_setup_retries_changed_optional_dependencies_without_blocking_launch"],
        globals())
