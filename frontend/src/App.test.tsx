import {cleanup,render,screen,waitFor} from '@testing-library/react';
import {afterEach,it,expect,vi} from 'vitest';
import App from './App';
vi.mock('./api',()=>({api:{list:vi.fn().mockResolvedValue([])}}));
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('shows engine names and device returned by health in the header',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({transcription_engine:'whisperx',diarization_engine:'pyannote',device:'cuda',app_env:'development'})}));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('whisperx / pyannote · cuda'));
});
it('does not claim a real engine when health is unavailable',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('Engine status unavailable'));
});
