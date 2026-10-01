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
