import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

# auto_error=False so a missing/malformed Authorization header gets our own
# 401 (with WWW-Authenticate, per RFC 6750).
_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    # Compare as bytes: hmac.compare_digest raises TypeError on non-ASCII
    # str inputs, which would turn a garbage token into a 500 instead of 401.
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode(), settings.api_key.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
