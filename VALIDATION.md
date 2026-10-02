# Validation

## CPU-first pipeline (October 2, 2026)

Machine: Windows 11 laptop, Intel i7-13620H (6 performance + 4 efficiency cores,
16 threads), 16 GB RAM, no NVIDIA GPU, `NUM_THREADS=10` (physical-core default).

Tests: 64 backend tests pass in a quick run (`SKIP_MODEL_TESTS=1`), plus 5 real
sherpa-onnx inference tests that pass with downloaded models; 13 frontend tests pass.
Skipped: the MeetEval cross-check (MeetEval cannot be built here), the opt-in
WhisperX test, and a symlink test needing admin rights. The live app in local queue
mode reported `parakeet_onnx` / `sherpa_onnx` ready, created its database under
`%LOCALAPPDATA%\transcriber`, and completed a real job with stage timings.

SBC019 (Santa Barbara corpus, 21.8 min, 5 speakers), reference from `SBC019.cha`
via `eval.chat` (1,099 turns, 4,114 words):

| Engines | WER | DER | cpWER | Processing | Speed |
|---|---|---|---|---|---|
| parakeet_onnx + sherpa_onnx | 28.6% | 76.2% | 105.8% | 3,535 s | 0.37x |
| parakeet_onnx + pyannote | skipped: PyTorch not installable (see below) | | | | |

The full run was partly on battery and shared the CPU with other test runs, so its
speed is pessimistic. Threshold clustering (0.5) produced 19 speakers for 5, which
drives DER and cpWER (extra speakers' words count as insertions). WER is inflated by
numerals and contractions (gonna / going to) against spoken-form references.

Uncontended on AC power, 5-minute excerpt (300-600 s, 5 reference speakers):
Parakeet 1.32x real time; sherpa-onnx diarization about 1x real time:

| Clustering | Speakers | DER |
|---|---|---|
| threshold 0.5 | 6 | 48.8% |
| threshold 0.7 | 2 | 52.0% |
| threshold 0.9 | 1 | 61.6% |
| oracle count (5) | 4 | 48.5% |

No threshold fixes diarization on this heavily overlapping family conversation, so
the default stays 0.5; it has not been tuned on the test file. End to end the CPU
pipeline runs at roughly 0.5-0.6x real time on this laptop, so a 3-hour hearing
takes about 5-6 hours, with diarization (mostly segmentation) as the larger half.
These are conversational-speech results; court recordings must be evaluated on
their own references.

Environment limits found: Windows Application Control blocked PyTorch's `c10.dll`
(so pyannote could not run even with a token; no HF_TOKEN was available anyway) and
a compiled build dependency of MeetEval; cpWER uses the built-in implementation,
checked against MeetEval's definition. Parakeet decoding was fastest with 6 threads
(the performance-core count) in a battery-powered benchmark; 10 or 16 were slower.

## Engine provenance and demo safeguards

Added regression coverage for persistent per-job engine metadata (including JSON
exports), executing worker settings, additive migration of legacy records,
production refusal for any mock configuration, startup logging without token
values, health response, mock/mixed-engine banners, historical warnings, and
the frontend health indicator. The real-engine duration integration test is opt-in
(`RUN_REAL_ENGINE_TEST=1`, `REAL_ENGINE_AUDIO`, authorized `HF_TOKEN`), runs actual
WhisperX/pyannote, and requires duration within two seconds of ffprobe. It was not
run in this session because authorized real-model access is unavailable.
Verified after this change: 21 backend tests passed (one real-model integration
test skipped), 10 frontend tests passed, production TypeScript build passed,
Compose smoke test passed with persisted mock metadata, and a production/mock
worker container exited with the expected startup refusal message.

## Queue connectivity

The LAN frontend was verified against the dedicated Docker API endpoint after a
native/Docker split caused `503 Background queue unavailable`. Health now reports
one active `audio-transcriber` worker, and the LAN smoke test passed upload, Redis
enqueue, worker execution, transcript polling, rename, all exports, byte ranges,
and deletion. Backend tests cover ready/no-worker/unavailable health states,
credential-safe Redis endpoints, and visible enqueue errors. The UI disables upload
when Redis or the worker is unavailable.

## Port binding and liveness

Vite now binds to its configured port strictly, so a duplicate frontend exits with
an explicit port-in-use error instead of silently moving from 5173 to 5174. Compose
uses the dependency-free `/api/live` endpoint for API process health, while
`/api/health` continues to report Redis/RQ worker readiness. RQ worker discovery is
scoped directly to the configured queue and Redis checks have one-second socket
timeouts. Verified after this change: 22 backend tests passed (two integration
tests skipped), 11 frontend tests passed, the production frontend build passed,
and `docker compose config --quiet` passed.

Runtime recreation could not be completed during the final check because Docker
Desktop's WSL backend became stuck while starting and required an Administrator
service reset. Source validation above does not depend on that unavailable runtime.

## Windows storage cleanup

Local job deletion now handles nested artifacts, read-only files, and transient
Windows/OneDrive locks with bounded retries. It does not recreate missing folders
or follow a job-directory symlink. The full backend suite passed with 26 tests and
three optional tests skipped. Retrying the previously failed delete through the
running API returned HTTP 204 and removed the stale job record.

## Initial build

Verified on October 2, 2026 with `ENGINE=mock`:

- React strict TypeScript production build (`npm run build`).
- Three Vitest / React Testing Library tests: rendering, timestamp/word seeking,
  search highlighting/clearing, and speaker rename API submission.
- Backend pytest suite on Windows, using actual ffmpeg/ffprobe and a generated WAV.
- Full backend suite including optional metric tests in the Linux evaluation image.
- Evaluation CLI: mock inference, CSV output, saved predictions, and rescoring.
  Identical synthetic references produce zero WER/DER/cpWER; altered transcripts
  and speaker attribution produce nonzero metrics. These are correctness checks,
  not evidence of real transcription accuracy.
- `docker compose up --build` with the actual Redis/RQ worker, exercised by
  `scripts/smoke.py`: upload, completion, 40 turns / 3 speakers, rename, TXT/SRT/JSON
  downloads, original and normalized audio ranges, and deletion.
- Compose CPU and GPU override files validate with `docker compose ... config`.
- Frontend dependency audit: zero reported vulnerabilities at installation.

The local `.env` selects host ports 5174 and 8002 because existing applications use
5173 and 8000. Those applications were left running. Repository defaults remain
5173 and 8000. `.env` is ignored by Git and contains no token.

Not verified in this session: real WhisperX large-v3 / community-1 inference,
GPU image build and GPU memory offloading, real recording accuracy, and visual
browser inspection. No authorized Hugging Face token was available. The browser
runtime reported no available browser. See README for real-engine setup and the
evaluation workflow.
