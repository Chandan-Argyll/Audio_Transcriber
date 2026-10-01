import copy
import logging
import re
from pathlib import Path
from collections.abc import Iterator
from uuid import uuid4
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session
from app.config import get_settings
from app.models import Job, SessionLocal
from app.queue import queue, safe_redis_endpoint
from app.schemas import JobOut, Transcript
from app.storage import storage
from pipeline.exporters import export
from pipeline.preprocess import probe

router = APIRouter(prefix='/api/jobs', tags=['jobs'])
logger = logging.getLogger('thread.jobs')


def enqueue(job_id: str) -> None:
    settings = get_settings()
    try:
        queue().enqueue('pipeline.run.run_job', job_id, job_id=job_id, job_timeout=settings.job_timeout_sec)
    except Exception as exc:
        logger.exception(
            'Failed to enqueue job %s on queue %s at %s (%s)',
            job_id,
            settings.rq_queue_name,
            safe_redis_endpoint(settings.redis_url),
            type(exc).__name__,
        )
        raise


def find_job(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, 'Job not found')
    return job


@router.post('', response_model=JobOut, status_code=202)
def create_job(file: UploadFile = File(...), num_speakers: int | None = Form(None), min_speakers: int | None = Form(None), max_speakers: int | None = Form(None)) -> Job:
    options = {k: v for k, v in {'num_speakers': num_speakers, 'min_speakers': min_speakers, 'max_speakers': max_speakers}.items() if v is not None}
    if any(not 1 <= n <= 8 for n in options.values()) or (min_speakers or 1) > (max_speakers or 8) or (num_speakers is not None and (min_speakers is not None or max_speakers is not None)):
        raise HTTPException(400, 'Use an exact count OR a min/max range, between 1 and 8')
    job_id = str(uuid4())
    target = storage.path(job_id, 'original')
    size = 0
    try:
        with target.open('wb') as dest:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > get_settings().max_file_size_mb * 1024 * 1024:
                    raise HTTPException(413, 'File exceeds maximum upload size')
                dest.write(chunk)
        duration = probe(target)
    except ValueError as exc:
        storage.delete(job_id)
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        storage.delete(job_id)
        raise
    except Exception as exc:
        storage.delete(job_id)
        raise HTTPException(500, 'Media validation unavailable; check server ffprobe installation') from exc
    transcription, diarization = get_settings().active_engines()
    job = Job(id=job_id, filename=Path(file.filename or 'audio').name[:255], duration_sec=duration, options=options,
              transcription_engine=transcription, diarization_engine=diarization)
    with SessionLocal() as session:
        session.add(job)
        session.commit()
    try:
        enqueue(job.id)
    except Exception as exc:
        with SessionLocal() as session:
            failed = find_job(session, job.id)
            failed.stage, failed.error = 'failed', 'Queue unavailable. Please upload again.'
            session.commit()
        raise HTTPException(503, 'Background queue unavailable') from exc
    return job


@router.get('', response_model=list[JobOut])
def list_jobs() -> list[Job]:
    with SessionLocal() as session:
        return session.query(Job).order_by(Job.created_at.desc()).limit(30).all()


@router.get('/{job_id}', response_model=JobOut)
def get_job(job_id: str) -> Job:
    with SessionLocal() as session:
        return find_job(session, job_id)


@router.get('/{job_id}/transcript', response_model=Transcript)
def get_transcript(job_id: str) -> dict:
    with SessionLocal() as session:
        job = find_job(session, job_id)
        if job.stage == 'failed':
            raise HTTPException(500, job.error)
        if job.transcript is None:
            raise HTTPException(409, 'Transcript is still processing')
        return job.transcript


@router.patch('/{job_id}/speakers', response_model=Transcript)
def rename_speakers(job_id: str, labels: dict[str, str]) -> dict:
    with SessionLocal() as session:
        job = find_job(session, job_id)
        if job.transcript is None:
            raise HTTPException(409, 'Transcript is still processing')
        transcript = copy.deepcopy(job.transcript)
        valid = {s['label'] for s in transcript['speakers']} | {s['id'] for s in transcript['speakers']}
        if not labels or any(key not in valid or not value.strip() or len(value.strip()) > 60 for key, value in labels.items()):
            raise HTTPException(400, 'Unknown speaker or invalid label (1–60 characters)')
        for speaker in transcript['speakers']:
            speaker['label'] = labels.get(speaker['id'], labels.get(speaker['label'], speaker['label'])).strip()
        job.transcript = transcript
        session.commit()
        return transcript


def chunks(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open('rb') as source:
        source.seek(start)
        while length > 0:
            chunk = source.read(min(length, 65536))
            if not chunk:
                break
            length -= len(chunk)
            yield chunk


@router.get('/{job_id}/audio')
def audio(job_id: str, request: Request, normalized: bool = False) -> StreamingResponse:
    with SessionLocal() as session:
        find_job(session, job_id)
    # Preserve original by default; the UI requests WAV for browser compatibility.
    normalized_path = storage.path(job_id, 'normalized.wav')
    path = normalized_path if normalized and normalized_path.exists() else storage.path(job_id, 'original')
    if not path.exists():
        raise HTTPException(404, 'Audio file missing')
    size = path.stat().st_size
    start, end, status = 0, size - 1, 200
    headers = {'Accept-Ranges': 'bytes'}
    range_header = request.headers.get('range')
    if range_header:
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
        if not match or not any(match.groups()):
            raise HTTPException(416, 'Invalid byte range', headers={'Content-Range': f'bytes */{size}'})
        a, b = match.groups()
        start = int(a) if a else max(0, size - int(b))
        end = min(int(b), size - 1) if a and b else size - 1
        if start > end or start >= size:
            raise HTTPException(416, 'Range outside file', headers={'Content-Range': f'bytes */{size}'})
        status = 206
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
    headers['Content-Length'] = str(end - start + 1)
    return StreamingResponse(chunks(path, start, end - start + 1), status_code=status, headers=headers, media_type='audio/wav' if path == normalized_path else 'application/octet-stream')


@router.get('/{job_id}/export')
def export_job(job_id: str, format: str = 'txt') -> Response:
    if format not in {'txt', 'srt', 'json'}:
        raise HTTPException(400, 'Format must be txt, srt, or json')
    transcript = get_transcript(job_id)
    return Response(export(transcript, format), media_type='application/json' if format == 'json' else 'text/plain', headers={'Content-Disposition': f'attachment; filename="transcript-{job_id}.{format}"'})


@router.delete('/{job_id}', status_code=204)
def delete_job(job_id: str) -> Response:
    with SessionLocal() as session:
        job = find_job(session, job_id)
        if job.stage not in {'completed', 'failed'}:
            raise HTTPException(409, 'Wait for processing to finish before deleting')
        storage.delete(job_id)
        session.delete(job)
        session.commit()
    return Response(status_code=204)
