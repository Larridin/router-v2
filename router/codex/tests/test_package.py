import subprocess
import sys
from pathlib import Path


def test_package_exposes_version() -> None:
    import routerlab

    assert routerlab.__version__ == "0.1.0"


def test_installed_console_script_can_import_package() -> None:
    script = Path(sys.executable).parent / "routerlab"

    result = subprocess.run(
        [script, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "semantic-centroid model router" in result.stdout
