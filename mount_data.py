import logging
import os
from pathlib import Path

from SkiNet.Azure.azure_blob_mounter import AzureBlobMounter

logging.basicConfig(level=logging.INFO)

def main() -> None:

    base = os.getenv("DEFAULT_HOME")
    mount_path = None
    if base:
        mount_path = Path(base).expanduser() / "azure_blob_data"
    # mount data (pass mount_path if set)
    mounter = AzureBlobMounter(mount_path=mount_path) if mount_path is not None else AzureBlobMounter()
    try:
        mounter.mount()
    except Exception:
        logging.getLogger(__name__).exception("Mount failed")
    finally:
        # remove runtime config containing secret
        mounter._cleanup()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
