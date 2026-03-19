import argparse
import logging
from pathlib import Path

from SkiNet.Azure.azure_blob_mounter import AzureBlobMounter
from SkiNet.Utils.project_paths import BLOBFUSE2_CONFIG_PATH

logging.basicConfig(level=logging.INFO)

def mount_data(mount_path: Path, config_path: Path, is_azure_mount: bool) -> None:
    """

    Mount Azure Blob Storage using blobfuse2

    The checked-in blobfuse2 config file should contain only non-secret settings,
    such as account name, container name, and mount options.

    - If authentication is done using service principal, Azure credentials are
    provided via environment variables at runtime and injected into a temporary runtime config.
    - For managed identity, the original config is used directly if it is already set up for MSI,
    or a temp copy with mode switched to MSI is created if the original config is set up for SPN.

    :param mount_path: Path to directory where Azure Blob Storage will be mounted on VM host or local.
            The directory is created if it does not already exist.
    :param config_path: Path to a blobfuse2 YAML config file containing non-secret settings.
        Default is the checked-in config at SkiNet.Utils.project_paths.BLOBFUSE2_CONFIG_PATH.
    :param is_azure_mount: Whether this mount is for Azure Blob Storage.
            If True, managed identity will be used. If False, the mounter will do service principal authentication.
    """
    mounter = AzureBlobMounter(mount_path=mount_path, config_path=config_path, is_azure_mount=is_azure_mount)
    try:
        mounter.mount()
    except Exception:
        logging.getLogger(__name__).exception("Mount failed")
    finally:
        # remove runtime config containing secret
        mounter._cleanup()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount-path", type=Path, required=True,
                    help="Path to mount Azure Blob Storage on VM host or local."
                    "Must be writable before running this script.")
    ap.add_argument("--config-path", type=Path, default=BLOBFUSE2_CONFIG_PATH,
                    help="Path to blobfuse2 config file. Default is the checked-in config at SkiNet.Utils.project_paths.BLOBFUSE2_CONFIG_PATH.")
    ap.add_argument("--is-azure-mount", action="store_true",
                    help="Whether to use Azure Blob Storage mount.")
    args = ap.parse_args()
    mount_data(args.mount_path, args.config_path, args.is_azure_mount)
