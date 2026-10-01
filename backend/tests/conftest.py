import io
import os
import tempfile
import wave
from pathlib import Path

os.environ['DATABASE_URL'] = 'sqlite:///' + str(Path(tempfile.mkdtemp()) / 'test.db')
os.environ['STORAGE_DIR'] = tempfile.mkdtemp()
os.environ['MOCK_DELAY_SEC'] = '0'

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models import Base, engine


@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\0\0' * 16000)
    return buffer.getvalue()
