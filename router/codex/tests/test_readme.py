from __future__ import annotations

import re
import shlex
from pathlib import Path


def test_readme_verify_commands_are_recognized() -> None:
    readme = Path(__file__).parents[1] / "README.md"
    blocks = re.findall(r"```bash verify\n(.*?)```", readme.read_text(), flags=re.DOTALL)

    assert len(blocks) >= 3
    routerlab_commands = {
        "download",
        "prepare",
        "train",
        "evaluate",
        "diagnose",
        "report",
        "run-all",
    }
    checked = 0
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("cd "):
                continue
            arguments = shlex.split(line)
            if arguments[:3] == ["uv", "run", "routerlab"]:
                assert arguments[3] in routerlab_commands
            else:
                assert arguments[0] in {"uv", "cargo", "go"}
            checked += 1
    assert checked >= 8
