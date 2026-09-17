import asyncio

import httpx

from multi_agent_asr.api.app import app


async def smoke_test() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            response.raise_for_status()
            print(response.json())


def main() -> None:
    asyncio.run(smoke_test())


if __name__ == "__main__":
    main()
