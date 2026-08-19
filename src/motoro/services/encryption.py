"""Symmetric encryption for secrets at rest using Fernet.

Not per-user: the key is a server-side secret (``CoreSettings.encryption_key``),
the same key for every row core encrypts. That is what makes this portable —
ARES's embedding-credential lookup and its LLM bridge both had *per-user*
secrets to resolve, which core deliberately does not do (see
``services.credentials`` and ``memory.embedding``); this module never needed
a user in the first place.
"""

from __future__ import annotations

import logging
import threading

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from motoro.config import settings

logger = logging.getLogger(__name__)

_fernet: MultiFernet | Fernet | None = None
_fernet_lock = threading.Lock()


def _get_fernet() -> Fernet | MultiFernet:
    global _fernet  # noqa: PLW0603
    if _fernet is not None:
        return _fernet

    with _fernet_lock:
        # Double-checked locking: re-check after acquiring the lock.
        if _fernet is not None:
            return _fernet

        key = getattr(settings, "encryption_key", None) or ""
        if not key:
            raise RuntimeError(
                "encryption_key is not set. Generate one with: "
                "python -c 'from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())'"
            )

        # Comma-separated list supports key rotation: MultiFernet encrypts with
        # the first key and can decrypt with any of them, so an old key stays
        # valid for existing rows until they are rewritten under the new one.
        keys = [k.strip() for k in key.split(",") if k.strip()]
        _fernet = MultiFernet([Fernet(k.encode()) for k in keys]) if len(keys) > 1 else Fernet(keys[0].encode())
    return _fernet


def encrypt(plaintext: str) -> str:
    """Encrypt *plaintext*, returning a Fernet token (safe to store as text)."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet token produced by :func:`encrypt`.

    Raises ``InvalidToken`` if the token is malformed or was encrypted under a
    key not in the configured rotation set.
    """
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        logger.warning("encryption.decrypt_failed: token invalid or key rotated out")
        raise


def reset_for_testing() -> None:
    """Drop the cached Fernet instance so a test can install a different key."""
    global _fernet  # noqa: PLW0603
    with _fernet_lock:
        _fernet = None
