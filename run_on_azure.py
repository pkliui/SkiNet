#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("Running: %s", shlex.join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def require_command(cmd: str) -> None:
    result = subprocess.run(["which", cmd], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Required command not found: {cmd}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull and run SkiNet in a fresh Docker container on Azure compute")

    parser.add_argument("--image", required=True, help="Docker image to run, e.g. skinet:latest")
    parser.add_argument(
        "--host-repo",
        default="/mnt/batch/tasks/shared/LS_root/mounts/clusters/skinet-compute/code/Users/pavel.kliuiev/repos/SkiNet",
        help="Path to repo on the host VM",
    )
    parser.add_argument(
        "--container-repo",
        default="/workplace/SkiNet",
        help="Path where repo is mounted inside container",
    )
    parser.add_argument(
        "--branch",
        default="data_on_azure",
        help="Git branch to checkout and pull on host",
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python executable inside container",
    )
    parser.add_argument(
        "--main-script",
        default="main_script.py",
        help="Main Python script to run inside container",
    )
    parser.add_argument(
        "--script-args",
        default="--config /workplace/SkiNet/main_config.yaml --azure-data",
        help="Arguments passed to the main script inside container",
    )
    parser.add_argument(
        "--use-managed-identity",
        default="true",
        choices=["true", "false"],
        help="Whether to pass USE_MANAGED_IDENTITY into container",
    )
    parser.add_argument(
        "--azure-managed-identity-client-id",
        default="",
        help="Optional client ID for user-assigned managed identity",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image = args.image
    host_repo = Path(args.host_repo)
    container_repo = args.container_repo
    branch = args.branch
    python_bin = args.python_bin
    main_script = args.main_script
    script_args = args.script_args
    use_managed_identity = args.use_managed_identity
    azure_managed_identity_client_id = args.azure_managed_identity_client_id

    logger.info("IMAGE=%s", image)
    logger.info("HOST_REPO=%s", host_repo)
    logger.info("CONTAINER_REPO=%s", container_repo)
    logger.info("BRANCH=%s", branch)
    logger.info("MAIN_SCRIPT=%s", main_script)
    logger.info("USE_MANAGED_IDENTITY=%s", use_managed_identity)

    if not (host_repo / ".git").exists():
        raise RuntimeError(f"{host_repo} is not a git repository")

    require_command("docker")
    require_command("git")

    run(["git", "config", "--global", "--add", "safe.directory", str(host_repo)], check=False)

    logger.info("Updating repo on host")
    run(["git", "-C", str(host_repo), "fetch", "origin"])
    run(["git", "-C", str(host_repo), "checkout", branch])
    run(["git", "-C", str(host_repo), "pull", "--ff-only", "origin", branch])

    logger.info("Pulling Docker image %s", image)
    run(["docker", "pull", image])

    inner_cmd = f"""
set -Eeuo pipefail
cd {shlex.quote(container_repo)}
echo 'Inside container: repo status'
git status --short || true
echo 'Running main script'
{shlex.quote(python_bin)} {shlex.quote(main_script)} {script_args}
""".strip()

    cmd = [
        "docker", "run", "--rm", "-it",
        "--cap-add=SYS_ADMIN",
        "--device=/dev/fuse",
        "--security-opt", "apparmor:unconfined",
        "-e", f"USE_MANAGED_IDENTITY={use_managed_identity}",
        "--mount", f"type=bind,src={host_repo},dst={container_repo}",
        "-w", container_repo,
        "-t", image
    ]

    if azure_managed_identity_client_id:
        cmd.extend(["-e", f"AZURE_MANAGED_IDENTITY_CLIENT_ID={azure_managed_identity_client_id}"])

    cmd.extend([
        image,
        "bash", "-lc", inner_cmd,
    ])

    run(cmd)

    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
