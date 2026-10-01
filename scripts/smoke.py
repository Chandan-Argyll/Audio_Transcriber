"""Validate the running Compose stack with real Redis queue submission."""
import io
import os
import time
import wave
import httpx


def main() -> None:
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\0\0' * 16000 * 20)
    with httpx.Client(base_url=os.environ.get('APP_URL', 'http://localhost:5173'), timeout=30) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/health').json()['engine'] == 'mock'
        response = client.post('/api/jobs', files={'file': ('smoke-test.wav', audio.getvalue(), 'audio/wav')})
        response.raise_for_status()
        job_id = response.json()['id']
        seen = set()
        for _ in range(120):
            status = client.get(f'/api/jobs/{job_id}').json()
            seen.add(status['stage'])
            if status['stage'] in {'completed', 'failed'}:
                break
            time.sleep(.25)
        assert status['stage'] == 'completed', status
        transcript = client.get(f'/api/jobs/{job_id}/transcript').json()
        assert len(transcript['turns']) == 40
        assert len(transcript['speakers']) == 3
        assert client.patch(f'/api/jobs/{job_id}/speakers', json={'Person 1': 'Alice'}).status_code == 200
        for format in ('txt', 'srt', 'json'):
            response = client.get(f'/api/jobs/{job_id}/export?format={format}')
            assert response.status_code == 200 and 'Alice' in response.text
        response = client.get(f'/api/jobs/{job_id}/audio', headers={'Range': 'bytes=0-43'})
        assert response.status_code == 206 and response.content == audio.getvalue()[:44]
        assert client.get(f'/api/jobs/{job_id}/audio?normalized=true', headers={'Range': 'bytes=-100'}).status_code == 206
        assert client.delete(f'/api/jobs/{job_id}').status_code == 204
        print('Compose smoke test passed: upload > Redis > worker > transcript > rename > exports > Range > delete')
        print('Observed stages:', ', '.join(sorted(seen)))


if __name__ == '__main__':
    main()
