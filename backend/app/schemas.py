from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class Word(BaseModel):
    w: str
    start: float
    end: float
    score: float | None = None


class Speaker(BaseModel):
    id: str
    label: str
    color_index: int


class Turn(BaseModel):
    id: int
    speaker_id: str
    start: float
    end: float
    text: str
    words: list[Word]


class Transcript(BaseModel):
    job_id: str
    duration_sec: float
    speakers: list[Speaker]
    turns: list[Turn]


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    stage: str
    progress: int
    error: str | None
    duration_sec: float
    created_at: datetime


class RenameRequest(BaseModel):
    root: dict[str, str] = Field(default_factory=dict)
