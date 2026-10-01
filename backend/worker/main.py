from redis import Redis
import logging
from rq import Queue, SimpleWorker
from app.config import get_settings
from app.models import init_db


def main() -> None:
    settings = get_settings()
    settings.require_safe_worker()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    logging.getLogger('thread.worker').info('Active worker: %s; HF_TOKEN present=%s', settings.engine_info(), bool(settings.hf_token))
    init_db()
    connection = Redis.from_url(settings.redis_url)
    # SimpleWorker avoids RQ's per-job fork so cached models persist between jobs.
    SimpleWorker([Queue(connection=connection)], connection=connection).work()


if __name__ == '__main__':
    main()
