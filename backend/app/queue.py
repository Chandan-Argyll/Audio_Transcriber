import logging
import threading
from queue import SimpleQueue
from urllib.parse import urlsplit, urlunsplit

from redis import Redis
from rq import Queue, Worker

from app.config import get_settings
from app.models import Job, SessionLocal

logger = logging.getLogger('thread.queue')
_local_jobs: SimpleQueue[str] = SimpleQueue()
_local_thread: threading.Thread | None = None
_local_lock = threading.Lock()


def _local_loop() -> None:
    from pipeline.run import run_job
    while True:
        job_id = _local_jobs.get()
        try:
            run_job(job_id)
        except Exception:
            logger.exception('Local transcription job %s crashed', job_id)


def _ensure_local_worker() -> None:
    global _local_thread
    with _local_lock:
        if _local_thread is None or not _local_thread.is_alive():
            # Daemon thread: one job at a time, and Ctrl+C never waits for a long job.
            _local_thread = threading.Thread(target=_local_loop, name='local-transcriber', daemon=True)
            _local_thread.start()


def job_timeout(job_id: str) -> int:
    """Long hearings on CPU can run slower than real time; never time out before 3x duration."""
    settings = get_settings()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        duration = job.duration_sec if job else 0
    return max(settings.job_timeout_sec, int(duration * 3))


def submit_local(job_id: str) -> None:
    _ensure_local_worker()
    _local_jobs.put(job_id)


def resume_local_jobs() -> int:
    """Re-run jobs a previous local API process left unfinished (restart, reload, crash)."""
    if get_settings().queue_backend != 'local':
        return 0
    with SessionLocal() as session:
        pending = session.query(Job).filter(Job.stage.not_in(('completed', 'failed'))).order_by(Job.created_at).all()
        for job in pending:
            job.stage, job.progress, job.error = 'queued', 0, None
        session.commit()
        ids = [job.id for job in pending]
    for job_id in ids:
        submit_local(job_id)
    if ids:
        logger.info('Resumed %d unfinished job(s) on the local queue', len(ids))
    return len(ids)


def redis_connection() -> Redis:
    settings = get_settings()
    return Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=1,
        socket_timeout=1,
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
    if settings.queue_backend == 'local':
        return {'queue_status': 'ready', 'worker_count': 1, 'queue_name': 'local (in-process)', 'redis_endpoint': 'none'}
    connection = redis_connection()
    endpoint = safe_redis_endpoint(settings.redis_url)
    try:
        connection.ping()
        workers = Worker.all(connection=connection, queue=queue(connection))
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
