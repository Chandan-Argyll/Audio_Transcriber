from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models import init_db
from app.queue import queue_health
from app.routers.jobs import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title='Thread · Speaker-labeled transcription', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(','), allow_methods=['*'], allow_headers=['*'])
app.include_router(router)


@app.get('/api/live')
def live() -> dict[str, str]:
    """Process liveness only; deliberately has no Redis/model dependencies."""
    return {'status': 'ok'}


@app.get('/api/health')
def health() -> dict[str, str | int]:
    settings = get_settings()
    queue = queue_health()
    return {
        'status': 'ok' if queue['queue_status'] == 'ready' else 'degraded',
        'engine': settings.engine,
        **settings.engine_info(),
        'queue_status': queue['queue_status'],
        'worker_count': queue['worker_count'],
        'queue_name': queue['queue_name'],
    }
