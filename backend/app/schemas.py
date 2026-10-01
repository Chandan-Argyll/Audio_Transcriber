from datetime import datetime
from pydantic import BaseModel, ConfigDict, computed_field


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
    transcription_engine: str = 'unknown'
    diarization_engine: str = 'unknown'
    speakers: list[Speaker]
    turns: list[Turn]


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    transcription_engine: str
    diarization_engine: str
    stage: str
    progress: int
    error: str | None
    duration_sec: float
    created_at: datetime


    @computed_field
    @property
    def status(self) -> str:
        return self.stage if self.stage in {'queued', 'completed', 'failed'} else 'processing'
