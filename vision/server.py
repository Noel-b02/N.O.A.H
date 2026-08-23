import json
import os
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from insightface.app import FaceAnalysis
from pydantic import BaseModel

load_dotenv()

# CPU-only end to end (onnxruntime's CPUExecutionProvider) — deliberately,
# so this never contends with Ollama/image-generation for VRAM, and so this
# whole feature ships on the 8GB edition too, unlike the SDXL/Hunyuan3D-2
# work which genuinely needs a GPU. Confirmed live before committing to
# this library: deepface is completely broken on this machine's Python
# version (every published release has unresolvable dependency conflicts),
# and the classic face_recognition library needs a model package that
# isn't on PyPI at all. insightface + onnxruntime installed cleanly and
# produced a correct detection + embedding on the first real test.
FACE_MODEL_NAME = os.environ.get("VISION_FACE_MODEL", "buffalo_l")
CAMERA_INDEX = int(os.environ.get("VISION_CAMERA_INDEX", "0"))
SCAN_INTERVAL_SECONDS = float(os.environ.get("VISION_SCAN_INTERVAL_SECONDS", "4"))
# ArcFace/buffalo_l's typical practical operating point for cosine
# similarity on webcam-quality images — not a universal constant, hence an
# env var override rather than a hardcoded assumption.
MATCH_THRESHOLD = float(os.environ.get("VISION_MATCH_THRESHOLD", "0.4"))
# Long enough that stepping out of frame briefly (e.g. to grab coffee)
# doesn't retrigger a greeting; short enough that a genuinely new visit
# later the same day does.
REANNOUNCE_COOLDOWN_SECONDS = float(os.environ.get("VISION_REANNOUNCE_COOLDOWN_SECONDS", "300"))

FACES_FILE = os.path.join(os.path.dirname(__file__), "faces.json")

app = FastAPI(title="N.O.A.H. Vision Service")

print(f"[vision] Loading face model '{FACE_MODEL_NAME}' (CPU)...")
face_app = FaceAnalysis(name=FACE_MODEL_NAME, providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[vision] Face model loaded.")


def load_known_faces():
    try:
        with open(FACES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_known_faces(faces):
    with open(FACES_FILE, "w") as f:
        json.dump(faces, f)


known_faces = load_known_faces()
print(f"[vision] Loaded {len(known_faces)} enrolled face(s).")


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def best_match(embedding):
    best_name = None
    best_score = 0.0
    for entry in known_faces:
        score = cosine_similarity(embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_name = entry["name"]
    if best_name is not None and best_score >= MATCH_THRESHOLD:
        return best_name
    return None


# Shared between the background thread and the /pending-event handler —
# small, contained lock rather than a heavier queue library, matching this
# codebase's preference for small direct pieces.
_state_lock = threading.Lock()
_pending_events = []
_current_identity = None  # None | "unknown" | a known name
_last_announced = {}  # name-or-"unknown" -> unix timestamp


def capture_one_frame():
    # Opened and released per tick, not held open continuously — keeps the
    # camera's hardware "in use" indicator light only flickering briefly
    # per scan instead of staying lit the whole time this service runs, a
    # real (if partial) privacy mitigation for a feature that's already
    # off by default.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def scan_loop():
    global _current_identity
    while True:
        try:
            frame = capture_one_frame()
            identity = None
            if frame is not None:
                faces = face_app.get(frame)
                if faces:
                    # Multiple simultaneous people is real scope creep this
                    # feature doesn't need — just take the largest face.
                    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    matched_name = best_match(largest.embedding)
                    identity = matched_name if matched_name else "unknown"

            with _state_lock:
                if identity != _current_identity:
                    _current_identity = identity
                    if identity is not None:
                        now = time.time()
                        if now - _last_announced.get(identity, 0) >= REANNOUNCE_COOLDOWN_SECONDS:
                            _last_announced[identity] = now
                            if identity == "unknown":
                                _pending_events.append({"type": "unknown", "name": None})
                            else:
                                _pending_events.append({"type": "known", "name": identity})
        except Exception as e:
            print(f"[vision] Scan loop error (non-fatal, retrying next tick): {e}")

        time.sleep(SCAN_INTERVAL_SECONDS)


threading.Thread(target=scan_loop, daemon=True).start()


class EnrollRequest(BaseModel):
    name: str
    base64: str
    mimeType: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "known_face_count": len(known_faces)}


@app.get("/pending-event")
async def pending_event():
    with _state_lock:
        if _pending_events:
            return _pending_events.pop(0)
    return None


@app.post("/enroll")
async def enroll(req: EnrollRequest):
    import base64 as b64

    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    try:
        image_bytes = b64.b64decode(req.base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.") from e

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the attached image.")

    faces = face_app.get(frame)
    if not faces:
        # SCRFD (the buffalo_l detector) is tuned for typical webcam framing,
        # where a face occupies a modest fraction of the frame — confirmed
        # directly that it misses faces in a tight close-up crop (e.g. a
        # face filling most of a 1024x1024 photo), and that what fixes it is
        # shrinking the face-to-frame *ratio*, not the image's absolute
        # resolution (a uniform resize alone doesn't touch that ratio and
        # doesn't help). Padding the frame onto a larger blank canvas shrinks
        # the ratio without touching the face's actual pixel detail. Chat-
        # driven enrollment gets arbitrary user photos, not just webcam-
        # distance shots, so retry once padded before giving up.
        h, w = frame.shape[:2]
        padded = np.full((h * 2, w * 2, 3), 128, dtype=np.uint8)
        padded[h // 2:h // 2 + h, w // 2:w // 2 + w] = frame
        faces = face_app.get(padded)

    if not faces:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the attached image — try a clearer, front-facing photo."
        )

    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    known_faces.append({"name": name, "embedding": largest.embedding.tolist()})
    save_known_faces(known_faces)

    return {"success": True, "name": name}
