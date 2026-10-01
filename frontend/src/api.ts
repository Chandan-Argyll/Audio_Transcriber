import type { Job, Transcript } from './types';
async function request<T>(path:string, options?:RequestInit):Promise<T> {
  const response = await fetch(`/api/jobs${path}`, options);
  if (!response.ok) { const data = await response.json().catch(()=>({detail:'The server could not be reached'})); throw new Error(typeof data.detail==='string' ? data.detail : 'Please check your inputs'); }
  return response.status === 204 ? undefined as T : response.json();
}
export const api = {
  list:()=>request<Job[]>(''), job:(id:string)=>request<Job>(`/${id}`), transcript:(id:string)=>request<Transcript>(`/${id}/transcript`),
  rename:(id:string, speaker:string, label:string)=>request<Transcript>(`/${id}/speakers`, {method:'PATCH', headers:{'Content-Type':'application/json'},body:JSON.stringify({[speaker]:label})}),
  delete:(id:string)=>request<void>(`/${id}`,{method:'DELETE'}),
  upload:(file:File, options:Record<string,number>, progress:(n:number)=>void)=>new Promise<Job>((resolve,reject)=>{
    const xhr=new XMLHttpRequest(); const body=new FormData(); body.append('file',file); Object.entries(options).forEach(([k,v])=>body.append(k,String(v)));
    xhr.open('POST','/api/jobs'); xhr.upload.onprogress=e=>{if(e.lengthComputable) progress(Math.round(e.loaded/e.total*100));};
    xhr.onload=()=>{try{const data=JSON.parse(xhr.responseText); if(xhr.status>=200&&xhr.status<300)resolve(data);else reject(new Error(typeof data.detail==='string'?data.detail:'Upload failed'));}catch{reject(new Error('Unexpected server response'));}};
    xhr.onerror=()=>reject(new Error('Upload connection failed. Check that the backend is running and the frontend API proxy points to it.')); xhr.send(body);
  })
};
