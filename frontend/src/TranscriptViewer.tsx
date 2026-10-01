import type {Transcript} from './types';
export function TranscriptViewer({transcript,filename}:{transcript:Transcript;filename:string;onUpdate:(t:Transcript)=>void}){return <div className="basic-transcript"><h2>{filename}</h2>{transcript.turns.map(t=><p key={t.id}>{t.text}</p>)}</div>;}
