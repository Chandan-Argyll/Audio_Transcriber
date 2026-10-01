# Validation

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

The local `.env` selects host ports 5174 and 8001 because existing applications use
5173 and 8000. Those applications were left running. Repository defaults remain
5173 and 8000. `.env` is ignored by Git and contains no token.

Not verified in this session: real WhisperX large-v3 / community-1 inference,
GPU image build and GPU memory offloading, real recording accuracy, and visual
browser inspection. No authorized Hugging Face token was available. The browser
runtime reported no available browser. See README for real-engine setup and the
evaluation workflow.
