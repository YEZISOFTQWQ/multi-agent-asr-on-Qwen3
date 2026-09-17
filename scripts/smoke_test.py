"""在不加载模型权重的情况下验证 API 生命周期和健康接口。"""

import asyncio

import httpx

from multi_agent_asr.api.app import app


async def smoke_test() -> None:
    """启动完整应用生命周期并检查健康接口。"""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            response.raise_for_status()
            print(response.json())


def main() -> None:
    """运行异步 smoke test。"""
    asyncio.run(smoke_test())


if __name__ == "__main__":
    main()
