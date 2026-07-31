"""SSRF prevention utilities.

Every place in ARES that accepts a user-supplied URL and subsequently makes
an outbound HTTP request MUST call ``validate_outbound_url`` before opening
the connection.  This prevents Server-Side Request Forgery attacks that would
allow an attacker to:

- Reach cloud IMDS (e.g., 169.254.169.254) and steal credentials.
- Probe internal services on private RFC-1918 / link-local networks.
- Use ``file://`` or ``gopher://`` schemes to read local files or attack
  other services.

Usage::

    from agentic_core.security.ssrf_guard import validate_outbound_url, SSRFError

    validate_outbound_url(user_url)           # raises SSRFError on failure
    validate_outbound_url(user_url, require_https=True)  # also rejects plain http
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

__all__ = ["SSRFError", "validate_outbound_url"]

# Private / link-local / loopback IPv4 ranges that must never be contacted.
_BLOCKED_IPV4_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / AWS IMDS
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),  # shared address space (RFC 6598)
    ipaddress.ip_network("192.0.0.0/24"),  # IETF protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1
    ipaddress.ip_network("198.18.0.0/15"),  # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
    ipaddress.ip_network("240.0.0.0/4"),  # reserved
    ipaddress.ip_network("255.255.255.255/32"),  # broadcast
]

_BLOCKED_IPV6_NETWORKS = [
    ipaddress.ip_network("::1/128"),  # loopback
    ipaddress.ip_network("fc00::/7"),  # unique local
    ipaddress.ip_network("fe80::/10"),  # link-local
    ipaddress.ip_network("::/128"),  # unspecified
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped (blocks private IPv4 via IPv6)
]

# Schemes that are never allowed for outbound HTTP calls.
_BLOCKED_SCHEMES = frozenset(
    {
        "file",
        "gopher",
        "dict",
        "ldap",
        "ldaps",
        "ftp",
        "sftp",
        "smb",
        "tftp",
        "jar",
        "netdoc",
        "mailto",
        "javascript",
        "data",
    }
)


class SSRFError(ValueError):
    """Raised when a URL is rejected by the SSRF guard."""


def _is_private_ip(address: str) -> bool:
    """Return True if *address* resolves to a private/reserved IP."""
    try:
        addr = ipaddress.ip_address(address)
    except ValueError:
        return False

    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in net for net in _BLOCKED_IPV4_NETWORKS)
    if isinstance(addr, ipaddress.IPv6Address):
        return any(addr in net for net in _BLOCKED_IPV6_NETWORKS)
    return False  # pragma: no cover


def _resolve_and_check(hostname: str) -> None:
    """DNS-resolve *hostname* and reject if any result is a private IP.

    We resolve here so an attacker cannot bypass the check by using a
    publicly-routable DNS name that secretly points at 169.254.169.254
    (DNS rebinding / SSRF via CNAMEs).

    This is a synchronous DNS lookup.  For the endpoints in question
    (federation peer creation, webhook registration) a brief sync DNS
    round-trip on the hot path is acceptable because:
    1.  These are infrequent configuration operations, not per-request.
    2.  The alternative (async resolution) requires ``aiodns`` or a
        thread-pool executor — adding complexity for marginal benefit.
    """
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFError(f"Cannot resolve hostname '{hostname}': {exc}") from exc

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip = str(sockaddr[0])
        if _is_private_ip(ip):
            raise SSRFError(
                f"URL hostname '{hostname}' resolves to a private/reserved address "
                f"({ip}) — outbound requests to internal networks are not permitted."
            )


def validate_outbound_url(
    url: str,
    *,
    require_https: bool = False,
    resolve_dns: bool = True,
    allow_private: bool = False,
) -> None:
    """Validate that *url* is safe to use as an outbound HTTP destination.

    Raises :class:`SSRFError` if the URL:

    - Uses a blocked scheme (``file://``, ``gopher://``, …).
    - Is missing a host.
    - Uses a non-https scheme when *require_https* is ``True``.
    - Has a hostname that is a bare private/reserved IP literal
      (unless *allow_private* is ``True``).
    - DNS-resolves to a private/reserved IP (when *resolve_dns* is ``True``
      and *allow_private* is ``False``).

    Parameters
    ----------
    url:
        The raw URL string supplied by the user.
    require_https:
        If ``True``, ``http://`` URLs are rejected in addition to blocked
        schemes.  Set this for endpoints where plaintext HTTP is unacceptable
        (e.g., webhook delivery in production).
    resolve_dns:
        If ``True`` (default), perform a synchronous DNS resolution and
        reject the URL if any resolved address is private.  Set to ``False``
        in unit tests that stub the network or when the caller already has a
        pre-validated IP.
    allow_private:
        If ``True``, skip the bare-IP and DNS-resolution private-network
        checks.  Use only in trusted dev/research deployments via
        ``AGENTIC_MCP_ALLOW_PRIVATE_URLS (or your product's prefix)=true``.  Scheme and host presence
        checks still apply.
    """
    if not url or not url.strip():
        raise SSRFError("URL must not be empty.")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if scheme in _BLOCKED_SCHEMES:
        raise SSRFError(
            f"URL scheme '{scheme}' is not permitted for outbound requests. Only 'http' and 'https' are accepted."
        )

    if scheme not in ("http", "https"):
        raise SSRFError(f"URL scheme '{scheme}' is not permitted. Only 'http' and 'https' are accepted.")

    if require_https and scheme != "https":
        raise SSRFError(f"Only HTTPS URLs are permitted for this endpoint. Received scheme: '{scheme}'.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL must contain a valid hostname.")

    if not allow_private:
        # Fast path: bare IP literal
        if _is_private_ip(hostname):
            raise SSRFError(
                f"URL hostname '{hostname}' is a private/reserved IP address — "
                "outbound requests to internal networks are not permitted."
            )

        # DNS resolution check (catches DNS rebinding / CNAME tricks)
        if resolve_dns:
            _resolve_and_check(hostname)
