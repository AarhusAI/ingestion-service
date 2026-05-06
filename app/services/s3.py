"""S3 object fetcher.

Treats whatever the deployment points us at (MinIO in dev, real AWS S3 in
prod, etc.) as generic S3-compatible storage. boto3 is configured with
``S3_ENDPOINT_URL`` / ``S3_ACCESS_KEY_ID`` / ``S3_SECRET_ACCESS_KEY`` /
``S3_REGION`` from settings.
"""

import logging
import tempfile
from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from app.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> BaseClient:
    """Lazily build the boto3 S3 client. Cached for the process lifetime."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        aws_access_key_id=settings.s3_access_key_id or None,
        aws_secret_access_key=settings.s3_secret_access_key or None,
        region_name=settings.s3_region,
        # Path-addressing works against both MinIO and AWS S3.
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def fetch_object_to_tempfile(bucket: str, key: str) -> str:
    """Download an S3 object to a NamedTemporaryFile. Returns the local path.

    Caller is responsible for ``os.unlink()`` on the returned path.
    Raises on transport / 404 / auth errors.
    """
    log.info("s3 fetch: bucket=%s key=%s", bucket, key)
    # delete=False is intentional — caller owns the file's lifetime and unlinks it
    # after the pipeline runs. Using a `with` block here would delete the file
    # before the pipeline can open it.
    fh = tempfile.NamedTemporaryFile(  # noqa: SIM115
        delete=False, suffix=_suffix_for_key(key)
    )
    try:
        _client().download_fileobj(bucket, key, fh)
        fh.flush()
        return fh.name
    except Exception:
        fh.close()
        try:
            import os

            os.unlink(fh.name)
        except OSError:
            pass
        raise
    finally:
        fh.close()


def _suffix_for_key(key: str) -> str:
    """Preserve the file extension so Tika / converters can dispatch by suffix."""
    if "." in key:
        return "." + key.rsplit(".", 1)[-1]
    return ""


def reset_client() -> None:
    """Test hook — drop the cached boto3 client so a new settings instance is picked up."""
    _client.cache_clear()
