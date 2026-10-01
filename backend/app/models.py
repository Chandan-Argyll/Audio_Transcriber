from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app.config import get_settings


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    stage: Mapped[str] = mapped_column(String(32), default='queued')
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    transcript: Mapped[dict | None] = mapped_column(JSON, nullable=True)


settings = get_settings()
if settings.database_url.startswith('sqlite:///'):
    Path(settings.database_url.removeprefix('sqlite:///')).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={'check_same_thread': False} if settings.database_url.startswith('sqlite') else {}, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
