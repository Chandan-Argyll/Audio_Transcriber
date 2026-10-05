import {cleanup,fireEvent,render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {afterEach,expect,it,vi} from 'vitest';
import App from './App';
import {api} from './api';
vi.mock('./api',()=>({api:{list:vi.fn().mockResolvedValue([]),job:vi.fn(),upload:vi.fn()}}));
afterEach(()=>{cleanup();vi.unstubAllGlobals();vi.clearAllMocks();});
const health={transcription_engine:'parakeet_onnx',diarization_engine:'sherpa_onnx',device:'cpu',app_env:'development',queue_status:'ready',worker_count:1,max_speakers:20,
  recording_presets:{general:{label:'General'},supreme_court:{label:'Supreme Court argument',min_speakers:10,max_speakers:14},meeting:{label:'Meeting',min_speakers:2,max_speakers:8}}};

it('supreme court preset fills the speaker range and is sent with the upload',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>health}));
  vi.mocked(api.upload).mockResolvedValue({id:'j1',filename:'argument.mp3',transcription_engine:'parakeet_onnx',diarization_engine:'sherpa_onnx',stage:'queued',progress:0,error:null,duration_sec:60,created_at:'2026-10-05T00:00:00Z'});
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('option',{name:'Supreme Court argument (10–14 speakers)'})).toBeInTheDocument());
  await userEvent.selectOptions(screen.getByLabelText(/Recording type/),'supreme_court');
  expect(screen.getByLabelText('Minimum')).toHaveValue(10);
  expect(screen.getByLabelText('Maximum')).toHaveValue(14);
  const input=document.querySelector('input[type=file]') as HTMLInputElement;
  fireEvent.change(input,{target:{files:[new File(['x'],'argument.mp3',{type:'audio/mpeg'})]}});
  await userEvent.click(screen.getByRole('button',{name:/Transcribe recording/}));
  await waitFor(()=>expect(api.upload).toHaveBeenCalled());
  expect(vi.mocked(api.upload).mock.calls[0][1]).toEqual({recording_type:'supreme_court',min_speakers:10,max_speakers:14});
});

it('general preset returns to automatic speaker detection',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>health}));
  render(<App/>);
  await waitFor(()=>expect(screen.getByRole('option',{name:'Meeting (2–8 speakers)'})).toBeInTheDocument());
  await userEvent.selectOptions(screen.getByLabelText(/Recording type/),'meeting');
  expect(screen.getByLabelText('Maximum')).toHaveValue(8);
  await userEvent.selectOptions(screen.getByLabelText(/Recording type/),'general');
  expect(screen.getByLabelText(/^Speakers/)).toHaveValue('auto');
  expect(screen.queryByLabelText('Maximum')).toBeNull();
});
