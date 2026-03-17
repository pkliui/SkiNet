import argparse
import logging
from pathlib import Path

from SkiNet.Azure.azure_blob_mounter import AzureBlobMounter

logging.basicConfig(level=logging.INFO)

def mount_data(mount_path: Path) -> None:
    mounter = AzureBlobMounter(mount_path=mount_path, is_azure_mount=True)
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
                    help="Path to mount Azure Blob Storage on VM host. Must be created and writable on the host before running this script.")
    args = ap.parse_args()
    mount_data(args.mount_path)
