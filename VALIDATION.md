# Validation

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
