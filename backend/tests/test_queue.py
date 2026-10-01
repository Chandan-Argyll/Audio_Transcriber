from types import SimpleNamespace

import pytest

from app import queue as queue_module
from app.config import get_settings
from app.routers import jobs


class RedisUp:
    def ping(self) -> bool:
        return True


def test_queue_health_ready_and_no_worker(monkeypatch):
    monkeypatch.setattr(queue_module, 'redis_connection', lambda: RedisUp())
    worker = SimpleNamespace(queues=[SimpleNamespace(name=get_settings().rq_queue_name)])
    monkeypatch.setattr(queue_module.Worker, 'all', lambda connection: [worker])
    assert queue_module.queue_health()['queue_status'] == 'ready'
    assert queue_module.queue_health()['worker_count'] == 1
    monkeypatch.setattr(queue_module.Worker, 'all', lambda connection: [])
    assert queue_module.queue_health()['queue_status'] == 'no_worker'


def test_queue_health_unavailable_hides_credentials(monkeypatch):
    class RedisDown:
        def ping(self):
            raise ConnectionError('offline')
    monkeypatch.setattr(queue_module, 'redis_connection', lambda: RedisDown())
    monkeypatch.setattr(get_settings(), 'redis_url', 'redis://user:secret@127.0.0.1:6380/4')
    status = queue_module.queue_health()
    assert status['queue_status'] == 'unavailable'
    assert status['redis_endpoint'] == 'redis://127.0.0.1:6380/4'
    assert 'secret' not in str(status)


def test_enqueue_failure_is_logged_with_safe_endpoint(monkeypatch, caplog):
    class BrokenQueue:
        def enqueue(self, *args, **kwargs):
            raise ConnectionError('offline')
    monkeypatch.setattr(jobs, 'queue', lambda: BrokenQueue())
    monkeypatch.setattr(get_settings(), 'redis_url', 'redis://user:secret@127.0.0.1:6380/4')
    with caplog.at_level('ERROR'), pytest.raises(ConnectionError):
        jobs.enqueue('job-id')
    assert 'redis://127.0.0.1:6380/4' in caplog.text
    assert 'secret' not in caplog.text
