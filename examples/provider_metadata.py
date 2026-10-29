"""Independent consumer, deterministic and offline: python examples/provider_metadata.py."""

import asyncio

import httpx

from motoro.services.provider_types import ProviderConfig
from motoro.services.providers import check_connection, discover_models, get_model_defaults, get_model_metadata


async def main() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "example/chat",
                        "name": "Example chat",
                        "supported_parameters": ["temperature"],
                        "default_parameters": {"temperature": 0, "stream": False, "stop": []},
                    }
                ]
            },
        )

    config = ProviderConfig(provider="openrouter", api_key="example-only")
    check = await check_connection(config, transport=httpx.MockTransport(respond))
    listing = await discover_models(config, transport=httpx.MockTransport(respond))
    metadata = get_model_metadata(model="example/chat", listing=listing)
    defaults = get_model_defaults(model=metadata.id, listing=listing)
    assert check.status == "ok" and listing.source == "api"
    assert defaults.temperature.known and defaults.temperature.value == 0
    assert defaults.additional["stream"].value is False
    assert not defaults.effort.known
    print(check.model_dump_json())
    print(metadata.model_dump_json())
    print(defaults.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
