from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app.config import get_settings


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    transcription_engine: Mapped[str] = mapped_column(String(32), default='unknown', server_default='unknown')
    diarization_engine: Mapped[str] = mapped_column(String(32), default='unknown', server_default='unknown')
    stage: Mapped[str] = mapped_column(String(32), default='queued')
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    # Deferred: a multi-hour hearing transcript is megabytes of JSON, and job
    # status polling and listings never need it.
    transcript: Mapped[dict | None] = mapped_column(JSON, nullable=True, deferred=True)


settings = get_settings()
if settings.database_url.startswith('sqlite:///'):
    Path(settings.database_url.removeprefix('sqlite:///')).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={'check_same_thread': False} if settings.database_url.startswith('sqlite') else {}, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    # Additive migration; old records must not inherit today's engine settings.
    with engine.begin() as connection:
        columns = {column['name'] for column in inspect(connection).get_columns('jobs')}
        missing = [name for name in ('transcription_engine', 'diarization_engine') if name not in columns]
        for name in missing:
            connection.execute(text(f"ALTER TABLE jobs ADD COLUMN {name} VARCHAR(32) NOT NULL DEFAULT 'unknown'"))
    if not missing:
        # Legacy transcripts were backfilled when the columns were added; avoid
        # loading every stored transcript on each startup.
        return
    with SessionLocal() as session:
        for job in session.query(Job).filter(Job.transcript.is_not(None)):
            if 'transcription_engine' not in job.transcript or 'diarization_engine' not in job.transcript:
                job.transcript = {**job.transcript, 'transcription_engine': job.transcription_engine, 'diarization_engine': job.diarization_engine}
        session.commit()
