import logging
import os

from azure.identity import ManagedIdentityCredential


def managed_identity_authentication() -> ManagedIdentityCredential:
    """
    Authenticate using Azure Managed Identity.
    """
    client_id = (
        os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID", "").strip()
        or os.getenv("DEFAULT_IDENTITY_CLIENT_ID", "").strip()
    )

    if client_id:
        logging.getLogger(__name__).info("Using user-assigned managed identity")
        return ManagedIdentityCredential(client_id=client_id)

    logging.getLogger(__name__).info("Using system-assigned managed identity")
    return ManagedIdentityCredential()
