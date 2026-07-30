"""User attribution.

Only ``UserSummary`` lives here. The ARES original bundled it with the HTTP auth
surface — ``UserRegister``, ``LoginRequest``, ``TokenResponse``, ``PasswordChange``,
``TokenCreate`` and friends — but those are two different concerns:

* ``UserSummary`` is an *attribution* DTO. Any resource that shows who owns it
  embeds one, which is why 13 modules import it, ``schemas.agent`` and
  ``schemas.pricing`` among them. Core needs it.
* The register/login/token schemas are a *transport contract* for an auth API.
  Whether authentication belongs to core at all is undecided, and core ships no
  HTTP layer to serve them from. They stay product-side until that is settled.

Splitting the file also drops a dependency: ``EmailStr`` — and with it
``email-validator`` — appeared only in the auth schemas. ``UserSummary`` types
``email`` as a plain ``str``, exactly as the original did.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class UserSummary(BaseModel):
    """Lightweight user reference for embedding in resource responses."""

    id: uuid.UUID
    display_name: str
    email: str

    model_config = {"from_attributes": True}
