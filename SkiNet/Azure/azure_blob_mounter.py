import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from SkiNet.Utils.project_paths import BLOBFUSE2_CONFIG_PATH


class AzureBlobMounter:
    """
    Mounts Azure Blob Storage using blobfuse2.

    The checked-in blobfuse2 config file should contain only non-secret settings,
    such as account name, container name, and mount options. Azure credentials are
    provided via environment variables at runtime and injected into a temporary runtime config.

    Example base blobfuse2.yaml:

    logging:
      type: syslog
      level: log_debug

    components:
      - libfuse
      - azstorage

    azstorage:
      type: block
      account-name: my-storage-account
      container: my-container
      endpoint: my-storage-account.blob.core.windows.net
      mode: spn
      clientid: '${AZURE_CLIENT_ID}'
      tenantid: '${AZURE_TENANT_ID}'
      clientsecret: '${AZURE_CLIENT_SECRET}'
    """

    def __init__(self,
                 mount_path: Path,
                 config_path: Path = BLOBFUSE2_CONFIG_PATH,
                 is_azure_mount: bool = True) -> None:
        """
        :param config_path: Path to a blobfuse2 YAML config file containing non-secret settings
        (account name, container, etc.) but not credentials.
        :param mount_path: Path to directory where Azure Blob Storage will be mounted.
            The directory is created if it does not already exist.
        :param is_azure_mount: Whether this mount is for Azure Blob Storage.
            If True, managed identity will be used. If False, the mounter will do service principal authentication.
        """
        self.is_azure_mount = is_azure_mount
        self.orig_cfg = Path(config_path)
        if not self.orig_cfg.exists():
            raise FileNotFoundError(f"blobfuse2 config not found: {self.orig_cfg}")
        logging.getLogger(__name__).info(f"Using blobfuse2 config: {self.orig_cfg}")

        self.mountpoint = mount_path
        self.runtime_cfg: Optional[Path] = None
        self.cfg_path: Optional[str] = None

        # ensure mountpoint exists
        self.mountpoint.mkdir(parents=True, exist_ok=True)

    def _ensure_unmounted(self) -> None:
        """
        Ensure the mountpoint is unmounted before attempting to mount.
        Otherwise Blobfuse2 may fail to mount properly.
        Raises RuntimeError if it still appears mounted after cleanup attempts.
        """
        logger = logging.getLogger(__name__)
        mp = str(self.mountpoint)

        def _is_mounted() -> bool:
            try:
                # check if mp dir is mounted using mountpoint utility (returns 0 if mounted)
                result = subprocess.run(
                    ["mountpoint", "-q", mp],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return True
            except FileNotFoundError:
                pass

            try:
                # check if mp dir appears in the /proc/mounts file listing active mounts (returns 0 if mounted)
                mounts = subprocess.run(
                    ["/bin/grep", mp, "/proc/mounts"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return mounts.returncode == 0
            except FileNotFoundError:
                return False

        if _is_mounted():
            logger.info("%s appears mounted — attempting cleanup", mp)
        else:
            logger.info("%s does not appear mounted, but attempting cleanup anyway", mp)

        # try unmounting with fusermount, then umount, then lazy umount
        for cmd in (
            ["fusermount", "-u", mp],
            ["umount", mp],
            ["umount", "-l", mp],
        ):
            try:
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Unmount succeeded with: %s", " ".join(cmd))
                    break
            except FileNotFoundError:
                continue

        if _is_mounted():
            raise RuntimeError(f"Mountpoint {mp} is still mounted; cannot continue")

    def _require_auth(self) -> None:
        """
        Ensure required environment variables are set in the environment
        """
        if not self.is_azure_mount:
            from SkiNet.Azure.azure_setup import AzureSetup
            AzureSetup.service_principal_authentication()
        else:
            pass

    def _create_runtime_config(self) -> Path:
        """
        Create a runtime configuration file for blobfuse2 with Azure credentials from environment variables.
        """
        # read original config and inline the secret and other credentials into a temp runtime config (chmod 600)
        cfg_data = yaml.safe_load(self.orig_cfg.read_text()) or {}
        if not isinstance(cfg_data, dict):
            raise ValueError(f"Invalid blobfuse2 config structure in {self.orig_cfg}")

        # inline Azure credentials into the config
        az = cfg_data.get("azstorage", {}) or {}

        # Managed identity: prefer using the original config.
        # If original config is SPN-mode, create a small temp copy switching mode to 'msi' (no secrets added).
        if self.is_azure_mount:
            if az.get("mode", "").lower() == "spn":
                az["mode"] = "msi"
                cfg_data["azstorage"] = az
                with tempfile.NamedTemporaryFile("w", delete=False, prefix="blobfuse2_runtime_msi_", suffix=".yaml") as tf:
                    self.runtime_cfg = Path(tf.name)
                    yaml.safe_dump(cfg_data, tf)
                os.chmod(self.runtime_cfg, 0o600)
                return self.runtime_cfg
            # original config already suitable for MSI — use it directly
            return self.orig_cfg

        # Service principal auth: require env vars and inline secrets into a temp runtime config
        required = ("AZURE_CLIENT_SECRET", "AZURE_TENANT_ID", "AZURE_CLIENT_ID")
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(f"Missing required Azure credentials for service principal auth: {', '.join(missing)}")

        az["clientsecret"] = os.environ["AZURE_CLIENT_SECRET"]
        az["tenantid"] = os.environ["AZURE_TENANT_ID"]
        az["clientid"] = os.environ["AZURE_CLIENT_ID"]
        cfg_data["azstorage"] = az

        # create a temporary runtime config file with the inlined credentials
        with tempfile.NamedTemporaryFile("w", delete=False, prefix="blobfuse2_runtime_", suffix=".yaml") as tf:
            self.runtime_cfg = Path(tf.name)
            yaml.safe_dump(cfg_data, tf)
        os.chmod(self.runtime_cfg, 0o600)  # set permissions to 600 so that only the current user can read it
        return self.runtime_cfg

    def _ensure_ownership(self) -> None:
        """
        Best-effort attempt to ensure mountpoint ownership so the runtime user can access files.
        Failure is logged but does not stop execution.
        """
        try:
            os.chown(self.mountpoint, os.getuid(), os.getgid())
        except PermissionError:
            logging.getLogger(__name__).warning(
                "Could not change ownership of mountpoint %s to %s:%s due to insufficient permissions",
                self.mountpoint,
                os.getuid(),
                os.getgid(),
            )
        except FileNotFoundError:
            logging.getLogger(__name__).warning(
                "Mountpoint %s does not exist when attempting to change ownership",
                self.mountpoint,
            )
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Failed to change ownership of mountpoint %s: %s",
                self.mountpoint,
                exc,
            )

    def _cleanup(self) -> None:
        """
        Clean up temporary files and resources.
        """
        if self.runtime_cfg and self.runtime_cfg.exists():
            try:
                self.runtime_cfg.unlink()
            except Exception:
                logging.getLogger(__name__).warning("Failed to remove runtime config %s", self.runtime_cfg)

    def mount(self) -> None:
        """
        Mount Azure Blob Storage to the specified mountpoint using blobfuse2 with a runtime config containing credentials from environment variables.
        """
        self._ensure_unmounted()
        self._require_auth()
        self._ensure_ownership()
        self.runtime_cfg = self._create_runtime_config()
        self.cfg_path = str(self.runtime_cfg)

        cmd = ["blobfuse2", "mount", str(self.mountpoint), f"--config-file={self.cfg_path}", "--log-level=LOG_DEBUG"]
        logging.getLogger(__name__).info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logging.getLogger(__name__).info("Mounted at %s", self.mountpoint)
        print("Mounted at", self.mountpoint)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--mount-path", type=Path, required=True,
                    help="Path to mount Azure Blob Storage on VM host. Must be created and writable on the host before running this script.")
    args = ap.parse_args()

    mounter = AzureBlobMounter(mount_path=args.mount_path)
    try:
        mounter.mount()
    except Exception:
        logging.getLogger(__name__).exception("Mount failed")
    finally:
        # remove runtime config containing secret
        mounter._cleanup()
