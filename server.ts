import express from 'express';
import * as fs from 'fs';
import { execFileSync, spawn, ChildProcess } from 'child_process';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434/api/generate";
const SPEECH_SERVICE_URL = process.env.SPEECH_SERVICE_URL ?? "http://localhost:5001";

const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL ?? "qwen3.5:4b";
const CODE_MODEL = process.env.OLLAMA_CODE_MODEL ?? "qwen3.5:9b";

const PERSONALITY_FILE = "personality.txt";
const MEMORY_FILE = "memories/memory.json";
const HISTORY_FILE = "memories/history/active.json";
const ARCHIVE_DIR = "memories/history/archive";

const MONTH_NAMES = [
  "january", "february", "march", "april", "may", "june",
  "july", "august", "september", "october", "november", "december"
];

const ALLOWED_FILES = new Set([
  "personality.txt",
  "memories/memory.json",
  "public/index.html"
]);

type Modification =
  | {
      action: "replace_file";
      file: string;
      content: string;
    }
  | {
      action: "replace_text";
      file: string;
      match: string;
      replace: string;
    }
  | {
      action: "append_file";
      file: string;
      content: string;
    }
  | {
      action: "set_json_value";
      file: string;
      path: string;
      value: any;
    };

interface CommitInfo {
  title: string;
  body?: string;
  author?: string;
}

interface ArchivedSession {
  archivedAt: string; // ISO timestamp of when this session was archived
  reason: "startup" | "manual";
  messages: string[];
}

// State to track if we have a draft on a feature branch
let pendingFiles: string[] = [];
let activeDraftBranch: string | null = null;

// Capture active.json's last-modified time BEFORE loading it, so a leftover
// session from a previous run gets archived under the date it actually
// happened on, not today's date.
const previousSessionEndedAt = (() => {
  try {
    return fs.statSync(HISTORY_FILE).mtime;
  } catch {
    return new Date();
  }
})();

let conversationHistory: string[] = loadHistory();
archiveSession("startup", previousSessionEndedAt);

let requestInFlight = false;
let pendingCommit: CommitInfo | null = null;

// Keeps the last retrieved archive context available for a few follow-up
// turns (e.g. "give me specifics", "quote it") so the model doesn't have to
// improvise from its own prior summary once the original recall query has
// scrolled out of isRecallQuery detection.
let stickyRecallContext = "";
let stickyRecallTurnsRemaining = 0;
const STICKY_RECALL_FOLLOWUP_TURNS = 3;


function loadHistory(): string[] {
  try {
    return JSON.parse(
      fs.readFileSync(HISTORY_FILE, "utf8")
    );
  } catch {
    return [];
  }
}

function saveHistory(): void {
  fs.writeFileSync(
    HISTORY_FILE,
    JSON.stringify(conversationHistory, null, 2)
  );
}

// Finds the next available session number for a given YYYY-MM-DD, so multiple
// sessions on the same day get date_1.json, date_2.json, etc.
function nextSessionNumberForDate(dateStr: string): number {
  const pattern = new RegExp(`^${dateStr}_(\\d+)\\.json$`);
  let max = 0;
  for (const file of listArchiveFiles()) {
    const match = path.basename(file).match(pattern);
    if (match) {
      const n = parseInt(match[1], 10);
      if (n > max) max = n;
    }
  }
  return max + 1;
}

// Archives whatever is currently in conversationHistory to its own dated
// file under ARCHIVE_DIR, then clears active.json for a fresh session.
// `archivedAt` lets the caller attribute the session to when it actually
// happened (e.g. active.json's last-modified time) rather than "now".
function archiveSession(reason: "startup" | "manual", archivedAt: Date = new Date()): string | null {
  if (conversationHistory.length === 0) return null;

  fs.mkdirSync(ARCHIVE_DIR, { recursive: true });

  const dateStr = archivedAt.toISOString().slice(0, 10); // YYYY-MM-DD
  const sessionNumber = nextSessionNumberForDate(dateStr);
  const archivePath = path.join(ARCHIVE_DIR, `${dateStr}_${sessionNumber}.json`);

  const session: ArchivedSession = {
    archivedAt: archivedAt.toISOString(),
    reason,
    messages: conversationHistory
  };

  fs.writeFileSync(archivePath, JSON.stringify(session, null, 2));

  conversationHistory = [];
  saveHistory();

  stickyRecallContext = "";
  stickyRecallTurnsRemaining = 0;

  console.log(`Archived session (${reason}) -> ${archivePath}`);
  return archivePath;
}

function listArchiveFiles(): string[] {
  try {
    return fs.readdirSync(ARCHIVE_DIR)
      .filter(f => f.endsWith(".json"))
      .map(f => path.join(ARCHIVE_DIR, f));
  } catch {
    return [];
  }
}

function loadArchive(filePath: string): ArchivedSession | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function findArchivesByDate(targetDate: Date): ArchivedSession[] {
  const matches: ArchivedSession[] = [];
  for (const file of listArchiveFiles()) {
    const session = loadArchive(file);
    if (!session) continue;
    const sessionDate = new Date(session.archivedAt);
    if (
      sessionDate.getFullYear() === targetDate.getFullYear() &&
      sessionDate.getMonth() === targetDate.getMonth() &&
      sessionDate.getDate() === targetDate.getDate()
    ) {
      matches.push(session);
    }
  }
  return matches;
}

function searchArchivesByKeyword(keyword: string): { date: string; line: string }[] {
  const results: { date: string; line: string }[] = [];
  const lowerKeyword = keyword.toLowerCase();
  for (const file of listArchiveFiles()) {
    const session = loadArchive(file);
    if (!session) continue;
    for (const line of session.messages) {
      if (line.toLowerCase().includes(lowerKeyword)) {
        results.push({ date: session.archivedAt, line });
      }
    }
  }
  return results;
}

// Looks for a DD/MM(/YYYY) numeric date or a "Month Day(, Year)" style date
// in a message. Numeric dates are parsed as DD/MM to match UK date order;
// year defaults to the current year when omitted.
function extractDateFromMessage(message: string): Date | null {
  const numeric = message.match(/\b(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{2,4}))?\b/);
  if (numeric) {
    const [, dayStr, monthStr, yearStr] = numeric;
    const day = parseInt(dayStr, 10);
    const month = parseInt(monthStr, 10) - 1;
    let year = yearStr ? parseInt(yearStr, 10) : new Date().getFullYear();
    if (yearStr && yearStr.length === 2) year += 2000;
    const d = new Date(year, month, day);
    if (!isNaN(d.getTime()) && month >= 0 && month <= 11) return d;
  }

  const monthPattern = new RegExp(
    `\\b(${MONTH_NAMES.join("|")})\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?\\b`,
    "i"
  );
  const monthMatch = message.match(monthPattern);
  if (monthMatch) {
    const monthIndex = MONTH_NAMES.indexOf(monthMatch[1].toLowerCase());
    const day = parseInt(monthMatch[2], 10);
    const year = monthMatch[3] ? parseInt(monthMatch[3], 10) : new Date().getFullYear();
    const d = new Date(year, monthIndex, day);
    if (!isNaN(d.getTime())) return d;
  }

  return null;
}

const RECALL_TRIGGER_PATTERN = /(what did (we|i) (talk|speak) about|what did (i|we) (say|discuss|mention)|do you remember (when|talking about|us talking about)|did we (talk|speak) about|have we (talked|spoken) about|what were we discussing)/i;

function extractTopicKeyword(message: string): string {
  const aboutMatch = message.match(/about\s+(.+?)[\?\.!]?$/i);
  if (aboutMatch) return aboutMatch[1].trim();
  return message.replace(RECALL_TRIGGER_PATTERN, "").trim();
}

function loadFile(filepath: string): string {
  try { return fs.readFileSync(filepath, 'utf8'); } catch { return ""; }
}

function writeFile(filepath: string, content: string): void {
  fs.writeFileSync(filepath, content, 'utf8');
}

function loadMemory(): string {
  try {
    return fs.readFileSync(MEMORY_FILE, "utf8");
  } catch (err) {
    console.error("Failed to load memory:", err);
    return "{}";
  }
}

function runGitCommand(args: string[]): string {
  try {
    return execFileSync('git', args, { stdio: 'pipe' }).toString().trim();
  } catch (error: any) {
    return error.stderr?.toString().trim() || error.message;
  }
}

// Git initialization on startup
const setupGit = () => {
  const currentBranch = runGitCommand(["branch", "--show-current"]);
  if (!currentBranch) {
    runGitCommand(["add", "."]);
    runGitCommand(["commit", "-m", "Initial web commit"]);
    runGitCommand(["branch", "-M", "master"]);
  }
};
setupGit();

// Warm the models so by the time the real request comes in, the model is already loaded and ready to respond quickly
fetch(OLLAMA_URL, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model: CHAT_MODEL, prompt: "hi", stream: false, options: { num_predict: 1 } })
}).catch(() => {});

// --- Speech service (Whisper + Kokoro) auto-start ---
const SPEECH_SERVICE_DIR = process.env.SPEECH_SERVICE_DIR ?? path.join(process.cwd(), "speech");
const SPEECH_SERVICE_PORT = process.env.SPEECH_SERVICE_PORT ?? "5001";
const SPEECH_SERVICE_AUTOSTART = (process.env.SPEECH_SERVICE_AUTOSTART ?? "true").toLowerCase() !== "false";

let speechServiceProcess: ChildProcess | null = null;

function getVenvPythonPath(): string {
  const isWindows = process.platform === "win32";
  return path.join(
    SPEECH_SERVICE_DIR,
    "venv",
    isWindows ? "Scripts" : "bin",
    isWindows ? "python.exe" : "python"
  );
}

async function isSpeechServiceRunning(): Promise<boolean> {
  try {
    const res = await fetch(`${SPEECH_SERVICE_URL}/health`, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}

async function startSpeechService(): Promise<void> {
  if (!SPEECH_SERVICE_AUTOSTART) {
    console.log("[speech] Auto-start disabled (SPEECH_SERVICE_AUTOSTART=false).");
    return;
  }

  if (await isSpeechServiceRunning()) {
    console.log("[speech] Speech service already running on its own — skipping auto-start.");
    return;
  }

  const pythonPath = getVenvPythonPath();

  if (!fs.existsSync(pythonPath)) {
    console.warn(
      `[speech] No venv Python found at ${pythonPath} — voice features will be unavailable ` +
      `until the speech service venv is set up (see speech/requirements.txt) or started manually.`
    );
    return;
  }

  console.log(`[speech] Starting speech service (${pythonPath})...`);

  speechServiceProcess = spawn(
    pythonPath,
    ["-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", SPEECH_SERVICE_PORT],
    { cwd: SPEECH_SERVICE_DIR, stdio: "inherit" }
  );

  speechServiceProcess.on("error", (err) => {
    console.warn(`[speech] Failed to start speech service: ${err.message}`);
    speechServiceProcess = null;
  });

  speechServiceProcess.on("exit", (code, signal) => {
    console.log(`[speech] Speech service process exited (code=${code}, signal=${signal})`);
    speechServiceProcess = null;
  });
}

function stopSpeechService(): void {
  if (speechServiceProcess && !speechServiceProcess.killed) {
    console.log("[speech] Stopping speech service...");
    speechServiceProcess.kill();
  }
}

startSpeechService();

app.get('/api/status', (_, res) => {
  res.json({
    activeDraftBranch,
    pendingFiles,
    historySize: conversationHistory.length
  });
});

// API: Speech-to-text — forwards recorded audio to the local Whisper service
// and returns the transcribed text. Uses express.raw() scoped to just this
// route since the body here is binary audio, not JSON.
app.post('/api/transcribe', express.raw({ type: '*/*', limit: '25mb' }), async (req, res) => {
  if (!req.body || req.body.length === 0) {
    return res.status(400).json({ error: "No audio data received." });
  }

  try {
    const contentType = req.headers['content-type'] || 'audio/webm';
    const blob = new Blob([req.body], { type: contentType });
    const formData = new FormData();
    formData.append('file', blob, 'audio.webm');

    const response = await fetch(`${SPEECH_SERVICE_URL}/transcribe`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Speech service returned ${response.status}: ${detail}`);
    }

    const data = await response.json() as { text: string; language?: string };
    res.json({ text: data.text });
  } catch (err: any) {
    console.error("Transcription failed:", err.message);
    res.status(500).json({ error: `Transcription failed: ${err.message}` });
  }
});

// API: Text-to-speech — forwards text to the local Piper service and streams
// back the resulting WAV audio.
app.post('/api/speak', async (req, res) => {
  const { text } = req.body;
  if (!text || !text.trim()) {
    return res.status(400).json({ error: "text is required." });
  }

  try {
    const response = await fetch(`${SPEECH_SERVICE_URL}/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Speech service returned ${response.status}: ${detail}`);
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    res.set('Content-Type', 'audio/wav');
    res.send(audioBuffer);
  } catch (err: any) {
    console.error("Speech synthesis failed:", err.message);
    res.status(500).json({ error: `Speech synthesis failed: ${err.message}` });
  }
});

// API: Send chat prompt to the assistant
app.post('/api/chat', async (req, res) => {
 const { message } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message is required." });
  }

  if (requestInFlight) {
    return res.status(429).json({ error: "Still processing the previous message — please wait." });
  }
  requestInFlight = true;

  const endSessionPattern = /^(end session|new session|clear session|archive session)\.?$/i;
  if (endSessionPattern.test(message.trim())) {
    const archivePath = archiveSession("manual");
    requestInFlight = false;
    return res.json({
      response: archivePath
        ? "Session archived. Starting fresh."
        : "Nothing to archive — this session is already empty.",
      hasProposedChanges: false
    });
  }

  conversationHistory.push(`User: ${message}`);
  saveHistory();

  const personality = loadFile(PERSONALITY_FILE);
  const memory = loadMemory();

  const wantsModification =/(modify|change|rewrite|update|edit|improve|refactor|remember|memorize|store|save)/i.test(message);

  console.log( "WANTS MODIFICATION:", wantsModification );

  const recallDate = extractDateFromMessage(message);
  const mentionsRecall = RECALL_TRIGGER_PATTERN.test(message);
  const isRecallQuery = !wantsModification && (
    mentionsRecall || (recallDate !== null && /what|talk|discuss|speak|say/i.test(message))
  );

  // A follow-up on an already-answered recall query (e.g. "give me specifics",
  // "quote it") — doesn't re-trigger isRecallQuery on its own, but should still
  // get the archived context and the smarter model.
  const isStickyRecallFollowup = !wantsModification && !isRecallQuery && stickyRecallTurnsRemaining > 0;

  console.log("IS RECALL QUERY:", isRecallQuery);
  console.log("IS STICKY RECALL FOLLOWUP:", isStickyRecallFollowup, "| turns remaining:", stickyRecallTurnsRemaining);

  // Short messages like ("hi", "thanks", "ok") are never complex on keyword grounds alone —
  // skip straight to the fast model rather than utilize the smart model to decrease latency
  // Modification requests still go to the smart model regardless of length, since
  // they need to reliably produce structured JSON output.
  const wordCount = message.trim().split(/\s+/).length;

  const isComplex = wantsModification || isRecallQuery || isStickyRecallFollowup || (
    wordCount > 5 && /(typescript|javascript|debug|refactor|git|branch)/i.test(message)
  );

  const selectedModel = isComplex? CODE_MODEL: CHAT_MODEL;

  const modificationTarget = /(remember|memory|memorise|memorize|store this|save this)/i.test(message) ? "memory.json": "personality.txt";

  const modificationInstructions = wantsModification ?
  `
  You are N.O.A.H., a self-modifying assistant.

  The user's requested modification target is: ${modificationTarget}

  CRITICAL: every response that includes "modifications" MUST also include a "commit" object at the top level, alongside "modifications" — not inside it. This applies to every single modification example below, with no exceptions.

  {
    "commit": {
      "title": "concise Conventional Commit title, e.g. feat(personality): ...",
      "body": "one sentence on why the change was made"
    },
    "modifications": [ ... ]
  }

  The following files may be modified:

  ${Array.from(ALLOWED_FILES).map(f => `- ${f}`).join("\n")}

  The modification actions work for ANY allowed file.

  Available actions:
  - replace_text
  - append_file
  - set_json_value

  Use replace_text when changing existing content.

  Example (note the required "commit" object):

  {
    "commit": {
      "title": "feat(personality): expand default detail level",
      "body": "User asked for more detail by default."
    },
    "modifications": [
      {
        "action": "replace_text",
        "file": "personality.txt",
        "match": "Be concise by default.",
        "replace": "Be concise by default, but use more detail when asked to explain something."
      }
    ]
  }

  Use append_file when adding new content without changing existing content.

  Example (note the required "commit" object):

  {
    "commit": {
      "title": "feat(personality): add casual tone instruction",
      "body": "User asked for a more casual tone."
    },
    "modifications": [
      {
        "action": "append_file",
        "file": "personality.txt",
        "content": "Keep things casual with the user."
      }
    ]
  }
  When using append_file:
  - content must contain ONLY the new text being added.
  - Do NOT include existing file contents.
  - Do NOT rewrite the file.
  - Do NOT repeat existing instructions.

  Prefer JSON modifications whenever possible.

  For JSON files, prefer set_json_value.

  Example (note the required "commit" object):

  {
    "commit": {
      "title": "chore(memory): update birthday fact",
      "body": "User corrected their stored birthday."
    },
    "modifications": [
      {
        "action": "set_json_value",
        "file": "memories/memory.json",
        "path": "facts.birthday",
        "value": "January 3rd"
      }
    ]
  }

  For large structural rewrites, you may still use:

  [UPDATE: filename]

  \`\`\`text
  complete file contents
  \`\`\`

  Rules:
  - Output a complete replacement file
  - Do NOT output UPDATE blocks for normal chat
  - Do NOT modify files unless explicitly requested
  - When modifying personality.txt, ONLY replace the contents of personality.txt.
  - Do NOT include System Instruction, CURRENT MEMORY, READ ONLY MEMORY CONTEXT, User Request, or any prompt text in the file.
  -  Do not modify existing instructions unless the user explicitly asks to change them.
  - Use append_file when adding a new instruction.
  - REMINDER: include the top-level "commit" object every time "modifications" is present, as shown in every example above.
  `
  : "";

  const metaSystemInstruction = (

    `The complete contents of personality.txt are:\n\n` + `${personality}\n\n` + 

    `READ ONLY MEMORY CONTEXT (not part of personality.txt):\n\n` + 
    `${memory}\n\n`   + 
    `CURRENT MEMORY is provided for reference.\n` +
    `Do not copy it into personality.txt.\n` +
    `Only modify memory.json when the user's requested modification target is memory.json.\n\n`+
    `The information in CURRENT MEMORY contains persistent facts and should be treated as true unless the user explicitly corrects them.\n\n` +

    modificationInstructions 
  
  );

  try {
    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, 60000);

    const recentHistory = conversationHistory.slice(-20).join("\n");

    let recallContext = "";
    if (isRecallQuery) {
      if (recallDate) {
        const sessions = findArchivesByDate(recallDate);
        const dateLabel = recallDate.toDateString();
        recallContext = sessions.length > 0
          ? `--- ARCHIVED CONVERSATION FROM ${dateLabel} ---\n` +
            sessions.map(s => s.messages.join("\n")).join("\n---\n") +
            `\n--- END ARCHIVED CONVERSATION ---\nAnswer the user's question using the archived conversation above.\n\n`
          : `--- NOTE: No archived conversation was found for ${dateLabel}. Tell the user you have no record of that day. ---\n\n`;
      } else {
        const topic = extractTopicKeyword(message);
        const hits = topic.length >= 3 ? searchArchivesByKeyword(topic).slice(0, 30) : [];
        recallContext = hits.length > 0
          ? `--- ARCHIVED MENTIONS OF "${topic}" ---\n` +
            hits.map(h => `[${new Date(h.date).toDateString()}] ${h.line}`).join("\n") +
            `\n--- END ARCHIVED MENTIONS ---\nAnswer the user's question using the archived mentions above.\n\n`
          : `--- NOTE: No archived mentions of "${topic}" were found. Tell the user you have no record of discussing that. ---\n\n`;
      }

      // Keep this context available for a few follow-up turns.
      stickyRecallContext = recallContext;
      stickyRecallTurnsRemaining = STICKY_RECALL_FOLLOWUP_TURNS;

      console.log("RECALL CONTEXT:");
      console.log(recallContext);
    } else if (isStickyRecallFollowup) {
      recallContext = stickyRecallContext.replace(
        "Answer the user's question using the archived conversation above.",
        "This is the same archived conversation retrieved for the user's earlier question — use it to answer this follow-up too. Do not invent details that aren't in it."
      ).replace(
        "Answer the user's question using the archived mentions above.",
        "These are the same archived mentions retrieved for the user's earlier question — use them to answer this follow-up too. Do not invent details that aren't in them."
      );
      stickyRecallTurnsRemaining--;

      console.log("USING STICKY RECALL CONTEXT, turns left after this:", stickyRecallTurnsRemaining);
    }

    const fullPrompt = wantsModification ? `System Instruction:\n${metaSystemInstruction}\n\n` + `User Request:\n${message}`: `System Instruction:\n${metaSystemInstruction}\n\n` + recallContext + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;
    
    console.log("WANTS MODIFICATION:", wantsModification);
    console.log("PROMPT SENT TO MODEL:");
    console.log(fullPrompt);

    console.log(`[MODEL] ${selectedModel} | complex=${isComplex}`);

    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: selectedModel,
        prompt: fullPrompt,
        stream: false,
        think: false,
        options: {
        num_predict: isComplex ? 3000 : 80,
        temperature: 0.3
        }
      })
    });

    if (!response.ok) throw new Error("Ollama connection failed");
    const data = await response.json() as { response: string };
    
    clearTimeout(timeout);
    
    const aiResponse = data.response;

    console.log("RAW AI RESPONSE:");
    console.log(aiResponse);
    console.log("RESPONSE LENGTH:", aiResponse?.length ?? 0);

    let jsonModifications: Modification[] = [];
    let isPureJsonResponse = false;

    try {
      const parsed = JSON.parse(aiResponse);
      isPureJsonResponse = true;

      pendingCommit = parsed.commit ?? null;

      if (Array.isArray(parsed.modifications)) {
          jsonModifications =
              parsed.modifications as Modification[];

          console.log(
              "JSON MODIFICATIONS FOUND:",
              jsonModifications.length
          );
      }
    } catch {
      // Not JSON, continue normally
    }
    
    if (!aiResponse.includes("I cannot execute") && !aiResponse.includes("I am an AI model")) {

      const isRefusal = aiResponse.includes("cannot execute") || aiResponse.includes("cannot modify") || aiResponse.includes("I am an AI model");

      if (!isRefusal) {
        const historySafeResponse = isPureJsonResponse
          ? `[Proposed modification: ${pendingCommit?.title ?? "changes drafted"}]`
          : aiResponse.replace(
              /\[UPDATE:.*?\]\s*```[\s\S]*?```/g,
              "[UPDATE GENERATED]"
            );

        conversationHistory.push(`Assistant: ${historySafeResponse}`);     
        saveHistory(); 
      }
    }

    if (conversationHistory.length > 40) {
      conversationHistory =
      conversationHistory.slice(-40);
  }

    // Parse out potential updates
    const pattern =  /\[UPDATE:\s*([^\]]+)\]\s*\n*\s*```[\w]*\n([\s\S]*?)```/g;
    const updates: { filepath: string; content: string }[] = [];
    let match;

    while ((match = pattern.exec(aiResponse)) !== null) {
      updates.push({
        filepath: match[1].trim(),
        content: match[2].trim()
      });
    }

    let hasProposedChanges = false;
    let gitDiff = "";

    if ( updates.length > 0 || jsonModifications.length > 0 ) {      
      
      // 1. Ensure clean workspace status before starting a draft branch
      const status = runGitCommand(["status", "--porcelain"]);

      const meaningfulChanges = status
        .split("\n")
        .filter(line =>
          line.trim() &&
          !line.includes("memories/history/active.json") &&
          !line.includes("memories/memory.json")
        );

      if (meaningfulChanges.length > 0) {
        return res.json({
          response: "I attempted to draft changes, but the local workspace has uncommitted files. Please resolve them first.",
          hasProposedChanges: false
        });
      }

      // 2. Checkout new draft branch
      const branchName = `feature/ai-${Date.now()}`;

      runGitCommand([ "checkout", "-b", branchName]);

      activeDraftBranch = branchName;      
      pendingFiles = [];

      // 3. Write drafts to disk
      for (const mod of jsonModifications) {

        if (!ALLOWED_FILES.has(mod.file)) {
          console.warn(
            `Blocked modification attempt: ${mod.file}`
          );
          continue;
        }

        if (mod.action === "replace_text") {

          const currentContent = loadFile(mod.file);

          const updatedContent = currentContent.replace( mod.match, mod.replace);

          writeFile( mod.file, updatedContent );

          pendingFiles.push(mod.file);
        }

        if (mod.action === "append_file") {

          const currentContent = loadFile(mod.file);

          console.log("BEFORE:");
          console.log(currentContent);

          const updatedContent = currentContent + "\n" + mod.content;

          console.log("AFTER:");
          console.log(updatedContent);

          writeFile(
            mod.file,
            updatedContent
          );

          pendingFiles.push(mod.file);
        }
      
        if (mod.action === "set_json_value") {

          const json = JSON.parse( loadFile(mod.file) );

          const keys = mod.path.split(".");

          let current = json;

          for (let i = 0; i < keys.length - 1; i++) {

            if (!current[keys[i]]) {
              current[keys[i]] = {};
            }

            current = current[keys[i]];
          }

          current[keys[keys.length - 1]] =  mod.value;

          writeFile(  mod.file, JSON.stringify(json, null, 2));

          pendingFiles.push(mod.file);
        }
      }

      

      for (const update of updates) {

        if (!ALLOWED_FILES.has(update.filepath)) {
          console.warn(
            `Blocked modification attempt: ${update.filepath}`
          );
          continue;
        }

        writeFile(update.filepath, update.content);
        pendingFiles.push(update.filepath);
      }

      hasProposedChanges = true;
      gitDiff = runGitCommand(["diff"]);
    }

    // Strip update tags out of conversational response text
    const cleanText = isPureJsonResponse
      ? ""
      : aiResponse.replace(/\[UPDATE:.*?\]\s*```[\s\S]*?```/g, "").trim();

    res.json({
      response: cleanText || (hasProposedChanges ? "I have drafted the requested changes for your review." : ""),
      commit: pendingCommit,
      hasProposedChanges,
      diff: gitDiff
    });

  } catch (err: any) {
    res.status(500).json({ error: err.message });
  } finally {
    requestInFlight = false;
  }
});

// API: Approve and Merge
app.post('/api/approve', (req, res) => {
  if (pendingFiles.length === 0) {
    return res.status(400).json({ error: "No pending modifications to approve." });
  }

  // Commit and merge sequence
console.log("ADDING:", pendingFiles);
console.log(runGitCommand(["add", ...pendingFiles]));

console.log("COMMITTING...");

const commitTitle = pendingCommit?.title ?? "chore(ai): self modification";

const commitBody = pendingCommit?.body ?? "No additional description provided.";

console.log(
  runGitCommand([
    "commit",
    "-m",
    commitTitle,
    "-m",
    `${commitBody}\n\nGenerated by N.O.A.H.`
  ])
);

// VS Code's Git extension reads .git/COMMIT_EDITMSG and shows it in the
// Source Control input box whenever that box is empty. Git already writes
// this file as part of `git commit -m`, but we write it explicitly here too
// so the exact title/body formatting is guaranteed regardless of git version.
try {
  fs.writeFileSync(
    path.join(".git", "COMMIT_EDITMSG"),
    `${commitTitle}\n\n${commitBody}\n\nGenerated by N.O.A.H.\n`
  );
} catch (err) {
  console.warn("Could not write .git/COMMIT_EDITMSG:", err);
}

console.log("CHECKOUT MASTER...");
console.log(
  runGitCommand(["checkout", "master"])
);

console.log("MERGING...");
console.log(
  runGitCommand([
    "merge",
    activeDraftBranch!
  ])
);

  if (activeDraftBranch) {
    runGitCommand([
      "branch",
      "-D",
      activeDraftBranch
    ]);
  }

  activeDraftBranch = null;
  pendingFiles = [];
  pendingCommit = null;
  res.json({ success: true, message: "Changes successfully merged to 'master'!" });
});

// API: Reject and Discard
app.post('/api/reject', (req, res) => {

  if (activeDraftBranch) {

    // Throw away any uncommitted changes
    runGitCommand(["reset", "--hard"]);

    // Go back to master
    runGitCommand(["checkout", "master"]);

    // Delete the draft branch
    runGitCommand([
      "branch",
      "-D",
      activeDraftBranch
    ]);
  }

  activeDraftBranch = null;
  pendingFiles = [];
  pendingCommit = null;

  res.json({
    success: true,
    message: "Changes discarded successfully."
  });
});

app.listen(PORT, () => {
  console.log(`N.O.A.H. server running at http://localhost:${PORT}`);
});

// Archive whatever's in the active session before the process actually exits,
// so Ctrl+C behaves the same as typing "end session" instead of leaving
// active.json sitting there until the next startup archives it late.
function shutdown(signal: string) {
  console.log(`\nReceived ${signal}, archiving session before exit...`);
  archiveSession("manual");
  stopSpeechService();
  process.exit(0);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));