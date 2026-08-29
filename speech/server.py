import io
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

import av
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro import KPipeline

load_dotenv()

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get(
    "WHISPER_COMPUTE_TYPE",
    "int8" if WHISPER_DEVICE == "cpu" else "float16",
)

KOKORO_LANG_CODE = os.environ.get("KOKORO_LANG_CODE", "b")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "bm_george")
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
KOKORO_USE_GPU = os.environ.get("KOKORO_USE_GPU", "false").lower() == "true"
KOKORO_SAMPLE_RATE = 24000

# Kokoro/torch pick up a GPU automatically if one is visible. Hiding it via
# CUDA_VISIBLE_DEVICES is a blunt but reliable way to force CPU regardless of
# which internal API Kokoro uses to pick its device.
if not KOKORO_USE_GPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

app = FastAPI(title="N.O.A.H. Speech Service")

print(f"[speech] Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
print("[speech] Whisper model loaded.")

print(f"[speech] Loading Kokoro (lang_code={KOKORO_LANG_CODE}, voice={KOKORO_VOICE}, gpu={KOKORO_USE_GPU})...")
kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
print("[speech] Kokoro pipeline loaded.")

print("[speech] Warming up Kokoro...")
try:
    _warmup_generator = kokoro_pipeline("hi", voice=KOKORO_VOICE, speed=KOKORO_SPEED)
    for _ in _warmup_generator:
        pass
    print("[speech] Kokoro warmup complete.")
except Exception as e:
    print(f"[speech] Kokoro warmup failed (non-fatal, first real request will be slower): {e}")


class SpeakRequest(BaseModel):
    text: str
    # "wav" (default, existing web UI behavior, untouched) or "ogg" — Ogg
    # container / Opus codec, the specific format Telegram's sendVoice API
    # requires for a real voice-message bubble (a WAV would need the
    # separate sendAudio API instead, losing that UX). Confirmed live via a
    # real encode-then-decode round trip before relying on it.
    format: str = "wav"


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"

    audio_bytes = await file.read()

    # A valid webm/opus recording of real speech is never this small — this
    # usually means the mic button was tapped rather than held, producing an
    # empty or header-only file that ffmpeg can't decode. Fail clearly here
    # instead of letting Whisper throw an "End of file" decoder error.
    if len(audio_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Recording too short or empty — hold the mic button longer while speaking."
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {"text": text, "language": info.language}
    except Exception as e:
        # Decoder failures (corrupt/empty audio) land here too, as a fallback
        # for anything the size check above didn't catch.
        raise HTTPException(
            status_code=400,
            detail="Could not decode the recorded audio. Try recording again."
        ) from e
    finally:
        os.remove(tmp_path)


@app.post("/speak")
async def speak(req: SpeakRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        # Kokoro splits long text into chunks and yields (graphemes, phonemes,
        # audio) per chunk — concatenate them into one continuous clip.
        generator = kokoro_pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
        audio_chunks = [audio for _, _, audio in generator]

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Kokoro produced no audio for this text.")

        full_audio = np.concatenate(audio_chunks)

        if req.format == "ogg":
            buffer = io.BytesIO()
            container = av.open(buffer, mode="w", format="ogg")
            # layout="mono" must be passed explicitly on both the stream and
            # the frame — confirmed live that omitting it silently produces
            # a stereo encode instead of erroring, which would double the
            # file size and desync duration for no benefit (Kokoro's output
            # is mono).
            stream = container.add_stream("libopus", rate=KOKORO_SAMPLE_RATE, layout="mono")
            audio_f32 = np.ascontiguousarray(full_audio, dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(audio_f32.reshape(1, -1), format="fltp", layout="mono")
            frame.sample_rate = KOKORO_SAMPLE_RATE
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):  # flush — required, or trailing packets are silently dropped
                container.mux(packet)
            container.close()
            return Response(content=buffer.getvalue(), media_type="audio/ogg")

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, KOKORO_SAMPLE_RATE, format="WAV")
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL_SIZE,
        "whisper_device": WHISPER_DEVICE,
        "kokoro_lang_code": KOKORO_LANG_CODE,
        "kokoro_voice": KOKORO_VOICE,
    }