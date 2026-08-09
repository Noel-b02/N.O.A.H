# Speech service setup (one-time)

The speech service (`speech/server.py`) handles voice input (Whisper) and voice output (Kokoro TTS). Requires Python 3.12.

1. Create and activate a virtual environment in `speech/`:

   ```bash
   cd speech
   python -m venv venv
   ```

   Windows: `venv\Scripts\activate` — macOS/Linux: `source venv/bin/activate`

2. Install torch **first, separately**, before anything else in `requirements.txt`. Do not skip this or install it alongside the rest of the file — `torch` is published under the same package name on both plain PyPI (CPU-only) and PyTorch's own index (CUDA-enabled), and mixing a CUDA `--index-url` into a normal `pip install -r requirements.txt` run is unreliable: pip's resolver can silently end up with either build depending on install order, so you can "successfully" install and still end up CPU-only with no error telling you so.

   **If you have an NVIDIA GPU** (recommended — Kokoro runs noticeably faster):

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```

   Check your driver supports CUDA 13.x first with `nvidia-smi` (look for the CUDA version in the header). If it's older, use the matching build from the index list at https://download.pytorch.org/whl/torch/ instead (e.g. `cu126`, `cu121`).

   **CPU only / no NVIDIA GPU:**

   ```bash
   pip install torch
   ```

3. Install everything else:

   ```bash
   pip install -r requirements.txt
   ```

4. Verify GPU acceleration actually took (if applicable):

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   This must print `True` for `KOKORO_USE_GPU=true` (in `speech/.env`) to actually do anything — if it prints `False`, Kokoro silently falls back to CPU with no error, it's just slower.

Once installed, `server.ts` auto-starts this service itself (see `startSpeechService()`) — there's no separate "run the speech service" step.
