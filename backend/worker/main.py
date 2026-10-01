from redis import Redis
from rq import Queue, Worker
from app.config import get_settings
from app.models import init_db


if __name__ == '__main__':
    init_db()
    connection = Redis.from_url(get_settings().redis_url)
    # Spawn worker supports Windows, keeps inference in one reusable process.
    # SimpleWorker avoids RQ's per-job fork so cached models persist between jobs.
    from rq import SimpleWorker
    SimpleWorker([Queue(connection=connection)], connection=connection).work()
