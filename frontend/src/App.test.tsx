import {cleanup,render,screen,waitFor} from '@testing-library/react';
import {afterEach,it,expect,vi} from 'vitest';
import App from './App';
vi.mock('./api',()=>({api:{list:vi.fn().mockResolvedValue([])}}));
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('shows engine names and device returned by health in the header',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({transcription_engine:'whisperx',diarization_engine:'pyannote',device:'cuda',app_env:'development',queue_status:'ready',worker_count:1})}));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('whisperx / pyannote · cuda'));
  expect(screen.getByRole('status')).toHaveTextContent('1 worker');
});
it('shows queue failure and disables transcription before upload',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({transcription_engine:'whisperx',diarization_engine:'pyannote',device:'cpu',app_env:'development',queue_status:'no_worker',worker_count:0})}));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('no worker'));
  expect(screen.getByRole('alert')).toHaveTextContent('no transcription worker is running');
  expect(screen.getByRole('button',{name:/Worker unavailable/})).toBeDisabled();
});
it('does not claim a real engine when health is unavailable',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('Engine status unavailable'));
});
it('explains a missing real engine and blocks upload',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({transcription_engine:'whisperx',diarization_engine:'pyannote',device:'cpu',app_env:'development',queue_status:'ready',worker_count:1,queue_backend:'local',engine_issue:'HF_TOKEN is missing',max_speakers:20})}));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('engine not ready'));
  expect(screen.getByRole('alert')).toHaveTextContent('HF_TOKEN is missing');
  expect(screen.getByRole('button',{name:/Engine not ready/})).toBeDisabled();
  expect(screen.getByRole('option',{name:/1–20 voices/})).toBeInTheDocument();
});
it('tells the user when the backend API is offline',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('backend API is not reachable'));
});
