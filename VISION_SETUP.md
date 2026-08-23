# Vision service setup (one-time)

The vision service (`vision/server.py`) gives Noah a camera: continuous background facial recognition, so he can notice when someone familiar (or unfamiliar) shows up and say something about it, plus chat-driven enrollment ("remember this face as X").

## Privacy — read this first

Unlike the speech service, **this is off by default** (`VISION_SERVICE_AUTOSTART=false`). Turning it on means your webcam is accessed continuously in the background, not just when you ask Noah a question — a materially more visible trust surface than an on-demand microphone. Nobody who clones this repo should get their webcam silently activated by a `git pull` and a restart.

What it actually does when enabled:
- Every couple of seconds (`VISION_SCAN_INTERVAL_SECONDS`, default 2), it opens the camera, grabs one frame, and immediately releases it — the camera isn't held open continuously, so the hardware "in use" light only flickers briefly per check rather than staying lit the whole time.
- Frames are processed locally and never saved to disk or sent anywhere — only a 512-number face embedding is stored (in `vision/faces.json`), for whoever you explicitly enroll.
- Identity isn't decided from a single frame — a match in any one of the last `VISION_RECOGNITION_WINDOW` ticks (default 5) counts, so one bad-angle frame doesn't erase an otherwise-solid recognition (confirmed live: without this, the same enrolled person could flicker between their name and "unknown" tick to tick). Enrolling several photos of the same person (different angles/lighting) helps this further — see "Enrolling a face" below.
- It only speaks up when someone's presence *changes* (nobody → somebody), and won't repeat itself for the same person within `VISION_REANNOUNCE_COOLDOWN_SECONDS` (default 300s / 5 minutes).

To enable it, set in `.env`:

```
VISION_SERVICE_AUTOSTART=true
```

## Setup

Requires Python (tested on 3.14; CPU-only, no GPU/CUDA needed).

1. Create a virtual environment in `vision/`:

   ```bash
   cd vision
   python -m venv venv
   ```

   Windows: `venv\Scripts\activate` — macOS/Linux: `source venv/bin/activate`

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This installs `insightface` + `onnxruntime` (CPU), chosen after testing ruled out the alternatives: `deepface` has unresolvable dependency conflicts on recent Python versions, and the classic `face_recognition` library depends on a model package that isn't published on PyPI at all. No `torch` here — this is deliberately CPU-only end to end, so it never competes with Ollama or image generation for VRAM.

3. First run downloads the `buffalo_l` face model (~280MB, cached under `~/.insightface/models/` afterward) — same one-time cost as Whisper/Kokoro's first-run model downloads for the speech service.

Once installed, `server.ts` auto-starts this service itself (see `startVisionService()`) if `VISION_SERVICE_AUTOSTART=true` — there's no separate "run the vision service" step.

## Enrolling a face

Attach a clear, front-facing photo in chat and say something like:

> remember this face as Noel

Noah will confirm once the face is detected and stored. A photo with no detectable face (too small, too dark, no face at all) gets a specific error asking for a clearer one instead of silently failing.

Enrollment doesn't overwrite or dedupe by name — saying "remember this face as X" again with a different photo just adds another reference sample for that name, and recognition matches against all of them. 3-5 photos covering a couple of angles and lighting conditions noticeably improves reliability over a single photo; there's no need to re-enroll from scratch each time.

## What "recognizing" someone actually means

This is a simple presence notifier, not an access-control system: Noah says something when a known or unknown face appears on camera. It doesn't gate any other feature, and match confidence is approximate — `VISION_MATCH_THRESHOLD` (default 0.4) trades off false matches against missed matches for typical webcam-quality video, not a guarantee of identity.
