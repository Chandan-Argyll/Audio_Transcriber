export type Stage = 'queued' | 'preprocessing' | 'transcribing' | 'aligning' | 'diarizing' | 'merging' | 'completed' | 'failed';
export interface Job {id:string; filename:string; transcription_engine:string; diarization_engine:string; stage:Stage; progress:number; error:string|null; duration_sec:number; created_at:string}
export interface Word {w:string; start:number; end:number; score:number|null}
export interface Speaker {id:string; label:string; color_index:number}
export interface Turn {id:number; speaker_id:string; start:number; end:number; text:string; words:Word[]}
export interface Transcript {job_id:string; duration_sec:number; transcription_engine:string; diarization_engine:string; speakers:Speaker[]; turns:Turn[]}
export interface Health {transcription_engine:string; diarization_engine:string; device:string; app_env:string; queue_status:'ready'|'no_worker'|'unavailable'; worker_count:number; queue_backend?:'redis'|'local'; engine_issue?:string|null; max_speakers?:number}
