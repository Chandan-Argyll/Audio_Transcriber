from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models import init_db
from app.routers.jobs import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title='Thread · Speaker-labeled transcription', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(','), allow_methods=['*'], allow_headers=['*'])
app.include_router(router)


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'engine': get_settings().engine}
