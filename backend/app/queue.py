import logging
from urllib.parse import urlsplit, urlunsplit

from redis import Redis
from rq import Queue, Worker

from app.config import get_settings

logger = logging.getLogger('thread.queue')


def redis_connection() -> Redis:
    settings = get_settings()
    return Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )


def queue(connection: Redis | None = None) -> Queue:
    settings = get_settings()
    return Queue(settings.rq_queue_name, connection=connection or redis_connection())


def safe_redis_endpoint(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname or 'unknown'
    if ':' in hostname:
        hostname = f'[{hostname}]'
    netloc = hostname
    if parts.port:
        netloc += f':{parts.port}'
    return urlunsplit((parts.scheme, netloc, parts.path, '', ''))


def queue_health() -> dict[str, str | int]:
    settings = get_settings()
    connection = redis_connection()
    endpoint = safe_redis_endpoint(settings.redis_url)
    try:
        connection.ping()
        workers = [
            worker for worker in Worker.all(connection=connection)
            if settings.rq_queue_name in {item.name for item in worker.queues}
        ]
        return {
            'queue_status': 'ready' if workers else 'no_worker',
            'worker_count': len(workers),
            'queue_name': settings.rq_queue_name,
            'redis_endpoint': endpoint,
        }
    except Exception as exc:
        logger.warning('Queue health check failed at %s (%s)', endpoint, type(exc).__name__)
        return {
            'queue_status': 'unavailable',
            'worker_count': 0,
            'queue_name': settings.rq_queue_name,
            'redis_endpoint': endpoint,
        }
