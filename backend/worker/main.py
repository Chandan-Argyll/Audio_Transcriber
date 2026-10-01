import logging
from rq import SimpleWorker
from app.config import get_settings
from app.models import init_db
from app.queue import queue, redis_connection, safe_redis_endpoint


def main() -> None:
    settings = get_settings()
    settings.require_safe_worker()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    logging.getLogger('thread.worker').info('Active worker: %s; HF_TOKEN present=%s', settings.engine_info(), bool(settings.hf_token))
    init_db()
    connection = redis_connection()
    endpoint = safe_redis_endpoint(settings.redis_url)
    try:
        connection.ping()
    except Exception as exc:
        raise RuntimeError(f'Worker cannot reach Redis at {endpoint}: {type(exc).__name__}') from exc
    logging.getLogger('thread.worker').info('Queue ready: %s at %s', settings.rq_queue_name, endpoint)
    # SimpleWorker avoids RQ's per-job fork so cached models persist between jobs.
    SimpleWorker([queue(connection)], connection=connection).work()


if __name__ == '__main__':
    main()
