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


__all__ = ["fetch_object_to_tempfile", "reset_client", "S3ObjectTooLarge"]


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


class S3ObjectTooLarge(Exception):
    """Raised when the S3 object's reported size exceeds ``max_upload_bytes``.

    Separate from the generic download path so the route layer can map it to
    HTTP 413 (rather than the 500 / S3_FETCH_FAILED bucket all other S3 errors
    fall into).
    """


def fetch_object_to_tempfile(bucket: str, key: str) -> str:
    """Download an S3 object to a NamedTemporaryFile. Returns the local path.

    Caller is responsible for ``os.unlink()`` on the returned path.
    Raises ``S3ObjectTooLarge`` if the object's ``ContentLength`` exceeds
    ``settings.max_upload_bytes``; raises on transport / 404 / auth errors
    otherwise.
    """
    log.info("s3 fetch: bucket=%s key=%s", bucket, key)

    # head_object before download so we don't stream a multi-GB object onto
    # local disk just to reject it. ContentLength is authoritative when the
    # bucket isn't using chunked / unknown-length uploads — which is the
    # normal case for Open WebUI's file storage.
    head = _client().head_object(Bucket=bucket, Key=key)
    size = head.get("ContentLength")
    max_bytes = settings.max_upload_bytes
    if isinstance(size, int) and size > max_bytes:
        raise S3ObjectTooLarge(
            f"S3 object size={size} exceeds max_upload_bytes={max_bytes} "
            f"(bucket={bucket} key={key})"
        )

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
