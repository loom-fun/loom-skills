"""Build subprocess argv for bundled Bash scripts, including Windows Git Bash."""
import os
from pathlib import Path
import shutil


def bash_executable():
    bash = shutil.which("bash")
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            for relative in ("bin/bash.exe", "usr/bin/bash.exe"):
                candidate = Path(git).parent.parent / relative
                if candidate.is_file():
                    bash = str(candidate)
                    break
        if bash and "windowsapps" in bash.lower():
            raise RuntimeError("Git Bash is required; WindowsApps bash.exe launches WSL")
    if not bash:
        raise RuntimeError("Bash not found; install Git Bash on Windows")
    return bash


def bash_script(script, *args):
    return [bash_executable(), os.fspath(script), *map(str, args)]


def bash_command(command):
    return [bash_executable(), "-c", command]
