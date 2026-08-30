import * as fs from 'fs';
import path from 'path';

// A standalone bridge, not part of server.ts's own request-handling cascade
// — its only coupling to the rest of the app is two outbound HTTP calls to
// its own /api/chat and /api/speak (the exact same way the web UI's own
// frontend talks to it), so it can't introduce a new way to corrupt that
// cascade's existing state (requestInFlight, conversationHistory, etc.). It
// inherits the existing single-flight 429 back-pressure and every intent
// branch for free, by construction.
// Deliberately NOT read at module-load time: server.ts imports this file
// before it calls dotenv.config() (a static `import` is hoisted above the
// importing file's own top-level code), so a top-level `const` here would
// always capture `undefined` regardless of what's actually in .env —
// confirmed live. Set inside startTelegramBot() instead, which only runs
// after dotenv.config() has already populated process.env.
let TELEGRAM_BOT_TOKEN: string | undefined;
let TELEGRAM_AUTHORIZED_CHAT_ID: string | undefined;
let TELEGRAM_API = "";
let TELEGRAM_FILE_API = "";
// Hardcoded to match server.ts's own PORT constant exactly (also a literal
// there, no env override anywhere in this codebase) — this is a loopback
// call to the same process, not a configurable external service.
const LOCAL_SERVER_URL = "http://localhost:3000";
const IMAGES_DIR = path.join(__dirname, "public", "generated-images");

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function tgSendMessage(chatId: number | string, text: string): Promise<void> {
  await fetch(`${TELEGRAM_API}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat_id: chatId, text })
  });
}

async function tgSendVoice(chatId: number | string, oggBuffer: Buffer): Promise<void> {
  const formData = new FormData();
  formData.append('chat_id', String(chatId));
  formData.append('voice', new Blob([new Uint8Array(oggBuffer)], { type: 'audio/ogg' }), 'voice.ogg');
  await fetch(`${TELEGRAM_API}/sendVoice`, { method: 'POST', body: formData });
}

async function tgSendPhoto(chatId: number | string, imagePath: string): Promise<void> {
  const imageBuffer = fs.readFileSync(imagePath);
  const formData = new FormData();
  formData.append('chat_id', String(chatId));
  formData.append('photo', new Blob([new Uint8Array(imageBuffer)]), path.basename(imagePath));
  await fetch(`${TELEGRAM_API}/sendPhoto`, { method: 'POST', body: formData });
}

async function tgGetFile(fileId: string): Promise<string> {
  const res = await fetch(`${TELEGRAM_API}/getFile?file_id=${fileId}`);
  const data = await res.json() as { ok: boolean; result?: { file_path: string }; description?: string };
  if (!data.ok || !data.result) throw new Error(`getFile failed: ${data.description}`);
  return data.result.file_path;
}

async function tgDownloadFile(filePath: string): Promise<Buffer> {
  const res = await fetch(`${TELEGRAM_FILE_API}/${filePath}`);
  return Buffer.from(await res.arrayBuffer());
}

// Long-poll timeout is Telegram's own server-side wait, not a client
// timeout — the client-side AbortSignal needs real headroom on top of it,
// or a slow-but-healthy long poll would get aborted right as it was about
// to return.
async function tgGetUpdates(offset: number, timeoutSec: number): Promise<any[]> {
  const res = await fetch(`${TELEGRAM_API}/getUpdates?offset=${offset}&timeout=${timeoutSec}`, {
    signal: AbortSignal.timeout((timeoutSec + 10) * 1000)
  });
  if (!res.ok) throw new Error(`getUpdates returned HTTP ${res.status}`);
  const data = await res.json() as { ok: boolean; result?: any[]; description?: string };
  if (!data.ok || !data.result) throw new Error(`getUpdates failed: ${data.description}`);
  return data.result;
}

// /api/transcribe expects a raw audio body (any container — it forwards to
// faster-whisper's PyAV backend, which sniffs the real format from magic
// bytes rather than trusting the filename/content-type) — confirmed live
// that a renamed Ogg/Opus file still decodes correctly through this same
// route the web UI's mic recordings already use.
async function transcribeViaLocalServer(audioBuffer: Buffer): Promise<string> {
  const res = await fetch(`${LOCAL_SERVER_URL}/api/transcribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'audio/ogg' },
    body: new Uint8Array(audioBuffer)
  });
  const data = await res.json() as { text?: string; error?: string };
  if (data.error) throw new Error(data.error);
  return data.text ?? "";
}

// Telegram retains undelivered updates for up to 24h and replays them once
// polling resumes — for a personal assistant that gets restarted often
// during normal use, silently answering hours-old messages late is a real
// correctness risk, not just noise (this codebase's own pendingPrintJob
// confirm/cancel flow is a concrete example of a command that must never
// fire late after a restart). So: on every boot, fetch whatever's pending
// with an immediate (timeout=0) call and advance past it without acting on
// any of it — deliberately drop the backlog rather than replay it.
async function drainBacklogOnBoot(): Promise<number> {
  try {
    const updates = await tgGetUpdates(0, 0);
    if (updates.length === 0) return 0;
    console.log(`[telegram] Dropping ${updates.length} backlog update(s) from while the server was offline.`);
    return updates[updates.length - 1].update_id + 1;
  } catch (e: any) {
    console.error("[telegram] Failed to drain backlog on startup (non-fatal):", e.message);
    return 0;
  }
}

// A few retries with a real delay between them — confirmed live that a
// generation request (image/3D) restarts the speech service via
// withGpuExclusive right as this call would land, and Whisper+Kokoro
// reloading takes a handful of seconds, not milliseconds.
async function fetchSpeechWithRetry(replyText: string, attempts = 4, delayMs = 2500): Promise<Response | null> {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(`${LOCAL_SERVER_URL}/api/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: replyText, format: 'ogg' })
      });
      if (res.ok) return res;
    } catch {
      // Speech service likely mid-restart — worth another attempt rather
      // than giving up on the first failure.
    }
    if (i < attempts - 1) await sleep(delayMs);
  }
  return null;
}

async function handleMessage(message: any): Promise<void> {
  const chatId = message.chat.id;

  // Unlike the local web UI (no auth, but only reachable on
  // localhost/LAN — a documented, accepted trade-off), a Telegram bot is
  // reachable by anyone who finds its username, and /api/chat can trigger
  // real destructive/costly actions (self-modification via git, Fusion 360
  // code execution, GPU-heavy generation). Bootstrap: until an authorized
  // chat_id is configured, the ONLY thing this bot ever does for anyone is
  // hand back their own chat_id and setup instructions — it never touches
  // /api/chat. Once configured, anyone else is silently ignored (no reply
  // at all, so a stranger can't even tell the bot is doing anything).
  if (!TELEGRAM_AUTHORIZED_CHAT_ID) {
    await tgSendMessage(
      chatId,
      `This is Noah's Telegram bridge, not yet claimed.\n\nYour chat ID is: ${chatId}\n\n` +
      `Add this to your .env file as:\nTELEGRAM_AUTHORIZED_CHAT_ID=${chatId}\n\n` +
      `Then restart the server to finish setup.`
    );
    return;
  }

  if (String(chatId) !== String(TELEGRAM_AUTHORIZED_CHAT_ID)) {
    console.log(`[telegram] Ignored message from unauthorized chat_id ${chatId}`);
    return;
  }

  let userMessage: string;
  let attachedImage: { base64: string; mimeType: string } | null = null;

  if (typeof message.text === "string") {
    userMessage = message.text;
  } else if (message.voice) {
    const filePath = await tgGetFile(message.voice.file_id);
    const audioBuffer = await tgDownloadFile(filePath);
    userMessage = await transcribeViaLocalServer(audioBuffer);
  } else if (message.photo) {
    // Telegram's `photo` array is sorted ascending by size — the last
    // entry is the largest available resolution.
    const largest = message.photo[message.photo.length - 1];
    const filePath = await tgGetFile(largest.file_id);
    const imageBuffer = await tgDownloadFile(filePath);
    attachedImage = { base64: imageBuffer.toString('base64'), mimeType: 'image/jpeg' };
    // /api/chat hard-requires a non-empty message, and several branches
    // (face-enroll, image-edit) key off regex-matching the message text,
    // not just an attached image being present — a neutral default avoids
    // accidentally tripping one of those when the user just wants Noah to
    // look at a photo, while a real caption ("remember this face as X")
    // still gets full parity with the web UI's text input.
    userMessage = message.caption || "Describe this image.";
  } else {
    await tgSendMessage(chatId, "I can only handle text, voice notes, and photos right now.");
    return;
  }

  const chatRes = await fetch(`${LOCAL_SERVER_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: userMessage, attachedImage, channel: 'telegram' })
  });
  const data = await chatRes.json().catch(() => ({ error: "Got an unreadable response from Noah's chat handler." })) as {
    response?: string;
    error?: string;
    generatedImageUrl?: string;
  };

  if (data.error) {
    await tgSendMessage(chatId, `Something went wrong: ${data.error}`);
    return;
  }

  const replyText = data.response || "(no response)";
  await tgSendMessage(chatId, replyText);

  // Voice reply is best-effort — a speech-service hiccup shouldn't erase
  // the text reply the user already has. Retries specifically because of a
  // real race confirmed live: an image-gen/edit or 3D-model request runs
  // through withGpuExclusive, which stops the speech service beforehand and
  // restarts it afterward — a /api/speak call landing in that restart
  // window (Whisper+Kokoro reloading, a few seconds) fails outright rather
  // than queuing, so this just waits it out instead of dropping the voice
  // reply on every generation request.
  try {
    const speakRes = await fetchSpeechWithRetry(replyText);
    if (speakRes) {
      await tgSendVoice(chatId, Buffer.from(await speakRes.arrayBuffer()));
    }
  } catch (e: any) {
    console.error("[telegram] Voice reply failed (non-fatal):", e.message);
  }

  // A near-free extension since the file path is already in the response
  // shape — 3D model (.glb) responses are left text-only, not meaningfully
  // viewable over Telegram.
  if (data.generatedImageUrl) {
    try {
      const imagePath = path.join(IMAGES_DIR, path.basename(data.generatedImageUrl));
      await tgSendPhoto(chatId, imagePath);
    } catch (e: any) {
      console.error("[telegram] Photo reply failed (non-fatal):", e.message);
    }
  }
}

async function pollLoop(): Promise<void> {
  let offset = await drainBacklogOnBoot();
  let backoffMs = 1000;

  while (true) {
    try {
      const updates = await tgGetUpdates(offset, 30);
      for (const update of updates) {
        offset = update.update_id + 1;
        if (update.message) {
          // Processed sequentially, not fire-and-forget — a burst of
          // messages should queue through /api/chat's existing single-
          // flight lock one at a time, same as if they'd all come from the
          // web UI in quick succession, rather than racing each other.
          try {
            await handleMessage(update.message);
          } catch (e: any) {
            console.error("[telegram] Error handling message:", e.message);
          }
        }
      }
      backoffMs = 1000;
    } catch (e: any) {
      console.error(`[telegram] getUpdates failed, retrying in ${backoffMs}ms:`, e.message);
      await sleep(backoffMs);
      backoffMs = Math.min(backoffMs * 2, 60000);
    }
  }
}

export function startTelegramBot(): void {
  // Read process.env here, not at module load — server.ts's `import` of
  // this file is hoisted above its own dotenv.config() call, so anything
  // read at module scope would always see an empty environment.
  TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
  TELEGRAM_AUTHORIZED_CHAT_ID = process.env.TELEGRAM_AUTHORIZED_CHAT_ID;

  if (!TELEGRAM_BOT_TOKEN) {
    console.log("[telegram] TELEGRAM_BOT_TOKEN not set — Telegram bridge disabled.");
    return;
  }
  TELEGRAM_API = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}`;
  TELEGRAM_FILE_API = `https://api.telegram.org/file/bot${TELEGRAM_BOT_TOKEN}`;

  if (!TELEGRAM_AUTHORIZED_CHAT_ID) {
    console.warn("[telegram] TELEGRAM_AUTHORIZED_CHAT_ID not set — bot will only reply with setup instructions until configured. See TELEGRAM_SETUP.md.");
  }
  console.log("[telegram] Starting long-poll loop...");
  pollLoop().catch(e => console.error("[telegram] Poll loop crashed unexpectedly:", e));
}
