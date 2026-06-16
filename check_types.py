"""
Convenience wrapper to check types using mypy

- Uses `mypy.ini` in the repo root.
- If no args are given, checks top-level *.py plus the files in SkiNet and Tests packages.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from shutil import which

from SkiNet.Utils.project_paths import REPO_ROOT

# default packages to check
DEFAULT_PKGS = ["SkiNet", "Tests"]
# path to .py files in the repo's root
DEFAULT_TOP_LEVEL = list(REPO_ROOT.glob("*.py"))


def _path_to_module(target: pathlib.Path) -> str | None:
    """
    Convert a repo-relative package file/directory path to a dotted module path.

    Returns None for top-level scripts that are not inside one of DEFAULT_PKGS.
    """
    try:
        rel_path = target.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None

    parts = rel_path.parts
    if not parts or parts[0] not in DEFAULT_PKGS:
        return None

    if target.is_file() and target.suffix == ".py":
        rel_path = rel_path.with_suffix("")

    return ".".join(rel_path.parts)


def check_types(targets: list[str], mypy_path: str) -> int:
    """
    Check types using mypy in given files or directories.
    Directories are checked in package mode (-p) to avoid duplicate-file issues.

    :return: Maximum return code across all targets. 0 = all succeed,
            non-zero if any target failed. The function does not stop on
            the first error; all targets are checked.
    """
    max_return_code = 0
    for idx, target in enumerate(targets):
        print(f"Checking types in {target}, item {idx+1}/{len(targets)}")
        path = pathlib.Path(target)

        if path.is_file():
            module_target = _path_to_module(path)
            if module_target is not None:
                mypy_target = ["-m", module_target]
            else:
                mypy_target = [str(path)]
        elif path.is_dir():
            # Use package mode to avoid duplicate-file issues mypy can raise on directories.
            # e.g. instead of directory SkiNet/ML/Model check module SkiNet.ML.Model.
            module_target = _path_to_module(path)
            if module_target is not None:
                mypy_target = ["-p", module_target]
            else:
                # Run rstrip in case someone specifies a backslash at the end, e.g. SkiNet/ML/Model/
                mypy_target = ["-p", target.rstrip(os.path.sep).replace(os.path.sep, ".")]
        else:
            raise FileNotFoundError(f"Target does not exist: {target}")

        cmd = [mypy_path, "--config-file", str(REPO_ROOT / "mypy.ini"), *mypy_target]
        proc = subprocess.run(cmd)
        max_return_code = max(max_return_code, proc.returncode)

    if max_return_code == 0:
        print("Mypy check passed for all targets!")
    else:
        print(f"Mypy finished with errors (return code {max_return_code})")

    return max_return_code


def main() -> int:
    """
    Check types using mypy. If no args are given, checks top-level *.py files and the files in DEFAULT_PKGS.
    """
    parser = argparse.ArgumentParser(description="Convenience wrapper to check types using mypy")
    parser.add_argument("-f", "--files", nargs="+", required=False, default=None,
                        help="Files or directories to check. If not specified, checks top-level *.py files and the files in DEFAULT_PKGS.")
    parser.add_argument("-m", "--path_to_mypy", required=False, default=None,
                        help="Path to mypy. If not specified, it will attempt to find mypy from PATH.")
    args = parser.parse_args()

    targets: list[str] = []
    if args.files:
        targets.extend(args.files)
    if not targets:
        targets = [str(p) for p in DEFAULT_TOP_LEVEL]
        targets.extend(DEFAULT_PKGS)

    mypy_path = args.path_to_mypy or which("mypy")
    if not mypy_path:
        raise RuntimeError("mypy executable not found on PATH; use --path_to_mypy to specify.")

    return check_types(targets, mypy_path)


if __name__ == "__main__":
    sys.exit(main())
