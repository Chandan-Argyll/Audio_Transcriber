from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models import init_db
from app.queue import queue_health, resume_local_jobs
from app.routers.jobs import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    resume_local_jobs()
    yield


app = FastAPI(title='Thread · Speaker-labeled transcription', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(','), allow_methods=['*'], allow_headers=['*'])
app.include_router(router)


@app.get('/api/live')
def live() -> dict[str, str]:
    """Process liveness only; deliberately has no Redis/model dependencies."""
    return {'status': 'ok'}


@app.get('/api/health')
def health() -> dict[str, str | int | None]:
    settings = get_settings()
    queue = queue_health()
    # Only a local backend runs inference in this process; with Redis the API
    # image has no model dependencies and the worker refuses to start instead.
    engine_issue = settings.engine_issue() if settings.queue_backend == 'local' else None
    return {
        'status': 'ok' if queue['queue_status'] == 'ready' and not engine_issue else 'degraded',
        'engine': settings.engine or 'cpu',
        **settings.engine_info(),
        'queue_status': queue['queue_status'],
        'worker_count': queue['worker_count'],
        'queue_name': queue['queue_name'],
        'queue_backend': settings.queue_backend,
        'engine_issue': engine_issue,
        'max_speakers': settings.max_speakers,
    }
