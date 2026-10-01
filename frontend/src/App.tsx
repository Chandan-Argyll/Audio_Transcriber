import {useCallback,useEffect,useRef,useState} from 'react';
import {api} from './api';
import {Icon} from './Icons';
import {TranscriptViewer} from './TranscriptViewer';
import type {Job,Transcript} from './types';
export const time=(seconds:number)=>{const s=Math.max(0,Math.floor(seconds));return [Math.floor(s/3600),Math.floor(s/60)%60,s%60].map(n=>String(n).padStart(2,'0')).join(':');};
const stages=['queued','preprocessing','transcribing','aligning','diarizing','merging','completed'];
const stageNames=['In queue','Preparing audio','Transcribing speech','Aligning words','Identifying speakers','Bringing it together','Ready to review'];
const stageProgress=[0,10,25,50,70,90,100];

export default function App(){
  const [file,setFile]=useState<File|null>(null),[recent,setRecent]=useState<Job[]>([]),[job,setJob]=useState<Job|null>(null),[transcript,setTranscript]=useState<Transcript|null>(null);
  const [busy,setBusy]=useState(false),[uploadProgress,setUploadProgress]=useState(0),[error,setError]=useState(''),[drag,setDrag]=useState(false),[exact,setExact]=useState(''),[min,setMin]=useState(''),[max,setMax]=useState(''),[mode,setMode]=useState('auto'),[mock,setMock]=useState(false);
  const input=useRef<HTMLInputElement>(null),selection=useRef(0);
  const refresh=useCallback(()=>api.list().then(setRecent).catch(()=>{}),[]);
  useEffect(()=>{void refresh();fetch('/api/health').then(r=>r.json()).then(d=>setMock(d.engine==='mock')).catch(()=>{});},[refresh]);
  const select=async(next:Job)=>{const token=++selection.current;setJob(next);setTranscript(null);setError('');if(next.stage==='completed')try{const value=await api.transcript(next.id);if(token===selection.current)setTranscript(value);}catch(e){if(token===selection.current)setError((e as Error).message);}};
  useEffect(()=>{
    if(!job || ['completed','failed'].includes(job.stage))return;
    let cancelled=false;const id=job.id;
    const poll=async()=>{try{const next=await api.job(id);if(cancelled)return;setJob(next);setError('');if(next.stage==='completed'){const value=await api.transcript(id);if(!cancelled)setTranscript(value);void refresh();}else if(next.stage==='failed')void refresh();}catch(e){if(!cancelled)setError((e as Error).message);}};
    const interval=window.setInterval(()=>void poll(),2000);return()=>{cancelled=true;clearInterval(interval);};
  },[job?.id,job?.stage,refresh]);
  const upload=async()=>{if(!file)return;setBusy(true);setError('');setUploadProgress(0);const options:Record<string,number>={};if(mode==='exact'&&exact)options.num_speakers=Number(exact);if(mode==='range'){if(min)options.min_speakers=Number(min);if(max)options.max_speakers=Number(max);}try{const next=await api.upload(file,options,setUploadProgress);await select(next);void refresh();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  const remove=async(item:Job)=>{try{await api.delete(item.id);if(job?.id===item.id){++selection.current;setJob(null);setTranscript(null);}void refresh();}catch(e){setError((e as Error).message);}};
  const processing=job&&!['completed','failed'].includes(job.stage);
  return <div className="app-shell">
    <header className="brand-bar"><a href="/" className="brand"><span className="brand-mark"><i/><i/><i/><i/></span>thread<span className="brand-divider"/><span className="brand-subtitle">EVERY VOICE, IN FOCUS</span></a><span className="local-badge"><span/> {mock?'Mock workspace':'Open-source transcription'}</span></header>
    <main className="workspace">
      <aside className="control-pane">
        <div className="eyebrow">YOUR WORKSPACE</div><h1>From conversation<br/>to clarity.</h1><p className="intro">A transcript for every voice. Upload a recording<br className="desktop-break"/> and let us untangle the conversation.</p>
        <section className="upload-card"><div className="section-heading"><h2>New transcription</h2><span className="step-tag">01</span></div>
          <input ref={input} type="file" className="sr-only" accept=".mp3,.wav,.m4a,.flac,.ogg,.mp4,.webm" onChange={e=>{setFile(e.target.files?.[0]||null);setError('');}}/>
          <div className={`drop-zone ${drag?'dragging':''} ${file?'has-file':'argyle'}`} onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);setFile(e.dataTransfer.files[0]||null);setError('');}}>
            <span className="upload-icon"><Icon name={file?'file':'upload'} size={27}/></span><strong>{file?file.name:'Drop your recording here'}</strong><p>{file?`${(file.size/1024/1024).toFixed(1)} MB · ${job?.filename===file.name?time(job.duration_sec):'Ready to upload'}`:'or choose a file from your computer'}</p><button className="browse-button" onClick={()=>input.current?.click()}>{file?'Change file':'Browse files'}</button><small>MP3, WAV, M4A, FLAC, OGG, MP4, WEBM</small>
          </div>
          <label className="field-label" htmlFor="speaker-mode">Speakers <span>OPTIONAL</span></label>
          <select id="speaker-mode" value={mode} onChange={e=>setMode(e.target.value)}><option value="auto">Detect automatically (1–8 voices)</option><option value="exact">I know the number of speakers</option><option value="range">Set a minimum and maximum</option></select>
          {mode==='exact'&&<label className="count-field">Number of speakers<input type="number" min="1" max="8" value={exact} onChange={e=>setExact(e.target.value)}/></label>}
          {mode==='range'&&<div className="range-fields"><label className="count-field">Minimum<input type="number" min="1" max="8" value={min} onChange={e=>setMin(e.target.value)}/></label><label className="count-field">Maximum<input type="number" min="1" max="8" value={max} onChange={e=>setMax(e.target.value)}/></label></div>}
          <button className="primary-button" disabled={!file||busy||!!processing} onClick={()=>void upload()}><Icon name="sound"/>{busy?'Uploading…':processing?'Transcription in progress':'Transcribe recording'}<span>↗</span></button>
          <p className="privacy-note"><span>◇</span> Open-source models. Your audio stays on your server.</p>
          {busy&&<div className="progress"><div style={{width:`${uploadProgress}%`}}/><span>Uploading {uploadProgress}%</span></div>}
          {error&&<div className="error" role="alert">{error}<button onClick={()=>void (job?select(job):upload())}>Retry</button></div>}
          {job&&<div className="stage-panel"><div className="section-heading"><h3>{job.stage==='completed'?'Transcript ready':job.stage==='failed'?'Processing failed':'Working on your recording'}</h3><span>{job.progress}%</span></div><div className="stage-progress"><div style={{width:`${job.progress}%`}}/></div><ol>{stages.map((stage,i)=>{const current=job.stage==='failed'?stageProgress.indexOf(job.progress):stages.indexOf(job.stage);const done=i<current||job.stage==='completed';const failed=job.stage==='failed'&&i===current;return <li key={stage} className={done?'done':failed?'failed':i===current?'current':''}><span>{done?<Icon name="check" size={12}/>:failed?<Icon name="close" size={12}/>:i+1}</span>{stageNames[i]}{i===current&&processing&&<b className="spinner"/>}</li>;})}</ol>{job.stage==='failed'&&<div className="error" role="alert">{job.error}<button disabled={!file} onClick={()=>void upload()}>Retry upload</button></div>}</div>}
        </section>
        <section className="recent-section"><div className="section-heading"><h2>Recent recordings</h2><span className="count-tag">{recent.length}</span></div>{recent.length===0?<p className="recent-empty">Your recordings will appear here.<br/>Start with your first conversation.</p>:recent.map(item=><div className={`recent-job ${job?.id===item.id?'selected':''}`} key={item.id}><button className="recent-load" onClick={()=>void select(item)}><span className="recent-icon"><Icon name="file"/></span><span><strong>{item.filename}</strong><small>{time(item.duration_sec)} <span>·</span> {new Date(item.created_at).toLocaleDateString()}</small></span><span className={`status-dot ${item.stage}`} title={item.stage}/></button>{['completed','failed'].includes(item.stage)&&<button className="icon-button delete-button" aria-label={`Delete ${item.filename}`} onClick={()=>void remove(item)}><Icon name="trash" size={15}/></button>}</div>)}</section>
        <footer className="pane-footer">MADE FOR THE MOMENTS THAT MATTER <span>✦</span></footer>
      </aside>
      <section className="viewer-pane">{transcript&&job?<TranscriptViewer key={job.id} transcript={transcript} filename={job.filename} onUpdate={setTranscript}/>:<><div className="viewer-placeholder-header"><span>TRANSCRIPT</span><span>WORDS, WITH CONTEXT</span></div>{processing?<div className="loading-view" aria-label="Loading transcript"><span className="spinner"/><h2>Every word is taking shape.</h2><p>{stageNames[stages.indexOf(job.stage)]} · {time(job.duration_sec)} recording</p>{[80,95,65,85,75].map((w,i)=><div className="skeleton" key={i} style={{width:`${w}%`}}/>)}</div>:<div className="empty-view"><div className="conversation-art"><div className="art-card art-back"><span/><span/><span/></div><div className="art-card art-front"><div className="art-avatar"/><div className="art-lines"><span/><span/><span/></div></div><span className="art-spark">✦</span><div className="art-wave">▂ ▅ ▃ ▇ ▄ ▂ ▆ ▃ ▅ ▂</div></div><div className="eyebrow">A SPACE FOR EVERY VOICE</div><h2>Your conversation,<br/>beautifully captured.</h2><p>Upload a recording to see who said what.<br/>Follow along, find the important bits, and take<br/>the conversation with you.</p><div className="empty-features"><span><Icon name="clock" size={16}/>Precise timestamps</span><span><Icon name="sound" size={16}/>Speaker labels</span><span><Icon name="download" size={16}/>Easy exports</span></div>{job?.stage==='failed'&&<div className="error" role="alert">{job.error}</div>}</div>}<div className="viewer-bottom-note"><span>✦</span> Less note-taking. More being present.</div></>}</section>
    </main>
  </div>;
}
