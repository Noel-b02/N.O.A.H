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

const SEARXNG_URL = process.env.SEARXNG_URL ?? "http://localhost:8080";

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

// Same idea, for web search: a follow-up like "give me the link to the sites
// you used" needs the same search results (with real URLs) that answered the
// original question, not just its own text summary from conversation history.
let stickySearchContext = "";
let stickySearchTurnsRemaining = 0;


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

// toISOString() always renders in UTC, which reads an hour off from wall-clock
// time during BST — this renders the same instant using local date/time parts
// plus the local UTC offset instead, so timestamps match what the clock said
// at the time while still round-tripping correctly through `new Date(...)`.
function toLocalISOString(date: Date): string {
  const pad = (n: number, len = 2) => String(n).padStart(len, "0");
  const offsetMin = -date.getTimezoneOffset();
  const sign = offsetMin >= 0 ? "+" : "-";
  const offH = pad(Math.floor(Math.abs(offsetMin) / 60));
  const offM = pad(Math.abs(offsetMin) % 60);
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}` +
    `${sign}${offH}:${offM}`
  );
}

// Archives whatever is currently in conversationHistory to its own dated
// file under ARCHIVE_DIR, then clears active.json for a fresh session.
// `archivedAt` lets the caller attribute the session to when it actually
// happened (e.g. active.json's last-modified time) rather than "now".
function archiveSession(reason: "startup" | "manual", archivedAt: Date = new Date()): string | null {
  if (conversationHistory.length === 0) return null;

  fs.mkdirSync(ARCHIVE_DIR, { recursive: true });

  const dateStr = `${archivedAt.getFullYear()}-${String(archivedAt.getMonth() + 1).padStart(2, "0")}-${String(archivedAt.getDate()).padStart(2, "0")}`; // local YYYY-MM-DD
  const sessionNumber = nextSessionNumberForDate(dateStr);
  const archivePath = path.join(ARCHIVE_DIR, `${dateStr}_${sessionNumber}.json`);

  const session: ArchivedSession = {
    archivedAt: toLocalISOString(archivedAt),
    reason,
    messages: conversationHistory
  };

  fs.writeFileSync(archivePath, JSON.stringify(session, null, 2));

  conversationHistory = [];
  saveHistory();

  stickyRecallContext = "";
  stickyRecallTurnsRemaining = 0;
  stickySearchContext = "";
  stickySearchTurnsRemaining = 0;

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

interface WebSearchResult {
  title: string;
  description: string;
  url: string;
}

// Some upstream engines SearXNG aggregates put highlight markup in their
// snippets — strip it so the model prompt gets plain text.
function stripHtmlTags(text: string): string {
  return text.replace(/<\/?[^>]+>/g, "");
}

async function webSearch(
  query: string,
  options: { categories?: string; timeRange?: string } = {}
): Promise<WebSearchResult[]> {
  try {
    let url = `${SEARXNG_URL}/search?q=${encodeURIComponent(query)}&format=json`;
    if (options.categories) url += `&categories=${encodeURIComponent(options.categories)}`;
    if (options.timeRange) url += `&time_range=${encodeURIComponent(options.timeRange)}`;

    const res = await fetch(url);

    if (!res.ok) {
      console.error("SearXNG search failed:", res.status, await res.text().catch(() => ""));
      return [];
    }

    const data = await res.json() as { results?: any[] };
    const results = data.results ?? [];

    return results.slice(0, 5).map((r: any) => ({
      title: stripHtmlTags(r.title ?? ""),
      description: stripHtmlTags(r.content ?? ""),
      url: r.url ?? ""
    }));
  } catch (err) {
    console.error("Web search error (is SearXNG running at " + SEARXNG_URL + "?):", err);
    return [];
  }
}

// "Give me today's headlines" is a fundamentally different task from a
// factual lookup like "who won the world cup" — it's not searching FOR
// anything specific, so a plain web search just surfaces static homepages
// (which rank highest for a bare term like "headlines"). SearXNG's news
// category plus a day-range filter routes through actual news-aggregator
// engines and returns real, dated, current articles instead.
async function fetchTodaysHeadlines(topic?: string): Promise<WebSearchResult[]> {
  return webSearch(topic?.trim() || "news", { categories: "news", timeRange: "day" });
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

// Old assistant replies from before the recall feature worked ("I don't have
// access...", "no record of that day", etc.) are now baked into the archives
// themselves. Feeding those back to the model as "context" causes it to
// parrot the same denial instead of answering — strip them out so recall
// context only contains real conversation, not the bug's own leftovers.
const STALE_NO_MEMORY_REPLY_PATTERN = /^Assistant:.*\b(don't have access to (our|any|the)|no record of|not in my (immediate )?memory|memory (starts fresh|is limited to|only (goes|holds))|no archived (conversation|mentions))\b/i;

function stripStaleNoMemoryReplies(messages: string[]): string[] {
  return messages.filter(line => !STALE_NO_MEMORY_REPLY_PATTERN.test(line));
}

// Spelled-out ordinals ("the first of August") alongside numeric ones
// ("1st of August" / "August 1st"). Longest names first so "twenty first"
// matches before the plain "first" inside it.
const ORDINAL_WORDS: Record<string, number> = {
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7,
  eighth: 8, ninth: 9, tenth: 10, eleventh: 11, twelfth: 12, thirteenth: 13,
  fourteenth: 14, fifteenth: 15, sixteenth: 16, seventeenth: 17, eighteenth: 18,
  nineteenth: 19, twentieth: 20, "twenty first": 21, "twenty second": 22,
  "twenty third": 23, "twenty fourth": 24, "twenty fifth": 25, "twenty sixth": 26,
  "twenty seventh": 27, "twenty eighth": 28, "twenty ninth": 29, thirtieth: 30,
  "thirty first": 31
};

const ORDINAL_WORD_PATTERN = Object.keys(ORDINAL_WORDS)
  .sort((a, b) => b.length - a.length)
  .map(w => w.replace(" ", "[\\s-]"))
  .join("|");

const DAY_TOKEN_PATTERN = `\\d{1,2}(?:st|nd|rd|th)?|${ORDINAL_WORD_PATTERN}`;

function parseDayToken(token: string): number | null {
  const numeric = token.match(/^(\d{1,2})(?:st|nd|rd|th)?$/i);
  if (numeric) return parseInt(numeric[1], 10);
  const normalized = token.toLowerCase().replace(/[\s-]+/g, " ").trim();
  return ORDINAL_WORDS[normalized] ?? null;
}

// Looks for a DD/MM(/YYYY) numeric date, a "Month Day(, Year)" style date, or a "Day (of) Month(, Year)" style date (digits or spelled-out ordinals) in a message. 
// Numeric dates are parsed as DD/MM to match UK date order; year defaults to the current year when omitted.
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

  const monthDayPattern = new RegExp(
    `\\b(${MONTH_NAMES.join("|")})\\s+(${DAY_TOKEN_PATTERN})(?:,?\\s+(\\d{4}))?\\b`,
    "i"
  );
  const monthDayMatch = message.match(monthDayPattern);
  if (monthDayMatch) {
    const monthIndex = MONTH_NAMES.indexOf(monthDayMatch[1].toLowerCase());
    const day = parseDayToken(monthDayMatch[2]);
    const year = monthDayMatch[3] ? parseInt(monthDayMatch[3], 10) : new Date().getFullYear();
    if (day !== null) {
      const d = new Date(year, monthIndex, day);
      if (!isNaN(d.getTime())) return d;
    }
  }

  const dayMonthPattern = new RegExp(
    `\\b(?:the\\s+)?(${DAY_TOKEN_PATTERN})\\s+(?:of\\s+)?(${MONTH_NAMES.join("|")})(?:,?\\s+(\\d{4}))?\\b`,
    "i"
  );
  const dayMonthMatch = message.match(dayMonthPattern);
  if (dayMonthMatch) {
    const day = parseDayToken(dayMonthMatch[1]);
    const monthIndex = MONTH_NAMES.indexOf(dayMonthMatch[2].toLowerCase());
    const year = dayMonthMatch[3] ? parseInt(dayMonthMatch[3], 10) : new Date().getFullYear();
    if (day !== null) {
      const d = new Date(year, monthIndex, day);
      if (!isNaN(d.getTime())) return d;
    }
  }

  return null;
}

const RECALL_TRIGGER_PATTERN = /(what did (we|i) (talk|speak) about|what did (i|we) (say|discuss|mention)|do you remember (when|talking about|us talking about)|(do|does|did) (you|we) remember (what|when|where|who|how)|did we (talk|speak) about|have we (talked|spoken) about|what were we discussing)/i;

// "Do/did you remember ...?" is someone asking us to recall something, not an
// instruction to store a new memory — unlike an imperative "remember that X".
const REMEMBER_QUESTION_PATTERN = /\b(do|does|did)\s+(you|we)\s+remember\b/i;

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

  const wantsModification = REMEMBER_QUESTION_PATTERN.test(message)
    ? /(modify|change|rewrite|update|edit|improve|refactor|memorize|store|save)/i.test(message)
    : /(modify|change|rewrite|update|edit|improve|refactor|remember|memorize|store|save)/i.test(message);

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

  // "Give me today's headlines" is a news-digest request, not a factual
  // lookup — it gets its own trigger and its own fetch path (SearXNG's news
  // category + day filter) rather than a plain web search, which just
  // surfaces static homepages for a bare term like "headlines".
  const HEADLINES_TRIGGER_PATTERN = /\b(headlines|breaking news|today'?s news|news today|on the news|what'?s (been )?reported|happening in the world)\b/i;
  const looksLikeHeadlinesQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && HEADLINES_TRIGGER_PATTERN.test(message);

  // Deterministic web-search trigger for common "needs current info" phrasings
  // — mirrors the recall-trigger approach above rather than relying on the
  // model to notice it needs to search and correctly emit [SEARCH: ...]. That
  // depends on smaller local models reliably following instructions, which in
  // practice they don't: they either wrap the tag in prose or skip it and
  // hallucinate an excuse instead. Catching the obvious cases here means the
  // search actually happens instead of being gated on model cooperation.
  const SEARCH_TRIGGER_PATTERN = /\b(latest news|who won|release date|launch date|premiere date|when('s| is| does| will).{0,30}(come out|coming out|releas(e|ing)|drop(ping)?|launch(ing)?|premier(e|ing))|current (weather|price|score|exchange rate)|how much (is|does|would)|price of|what'?s the weather|weather (today|forecast|right now))\b/i;
  const looksLikeSearchQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeHeadlinesQuery && SEARCH_TRIGGER_PATTERN.test(message);

  // A follow-up on an already-answered search or headlines request (e.g.
  // "give me the link to the sites you used", "which one said that") —
  // needs the same results (with real URLs) rather than the model
  // improvising from its own summary.
  const isStickySearchFollowup = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && stickySearchTurnsRemaining > 0;

  console.log("IS RECALL QUERY:", isRecallQuery);
  console.log("IS STICKY RECALL FOLLOWUP:", isStickyRecallFollowup, "| turns remaining:", stickyRecallTurnsRemaining);
  console.log("LOOKS LIKE HEADLINES QUERY:", looksLikeHeadlinesQuery);
  console.log("LOOKS LIKE SEARCH QUERY:", looksLikeSearchQuery);
  console.log("IS STICKY SEARCH FOLLOWUP:", isStickySearchFollowup, "| turns remaining:", stickySearchTurnsRemaining);

  // Short messages like ("hi", "thanks", "ok") are never complex on keyword grounds alone —
  // skip straight to the fast model rather than utilize the smart model to decrease latency
  // Modification requests still go to the smart model regardless of length, since
  // they need to reliably produce structured JSON output.
  const wordCount = message.trim().split(/\s+/).length;

  const isComplex = wantsModification || isRecallQuery || isStickyRecallFollowup || looksLikeHeadlinesQuery || looksLikeSearchQuery || isStickySearchFollowup || (
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

  // Not anchored to the whole string: smaller local models routinely ignore
  // "respond with EXACTLY this and nothing else" and wrap the tag in prose
  // anyway. Matching anywhere in the response catches it regardless, and the
  // surrounding prose gets discarded once we detect it (see below).
  const SEARCH_TAG_PATTERN = /\[SEARCH:\s*([^\]]+)\]/i;

  // Recall queries already have their own dedicated context (archived
  // conversations); the web-search tag is for outside-world facts the model
  // doesn't already know, so only offer it outside those two paths.
  const searchInstructions = (!wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !isStickySearchFollowup) ? `
  If answering requires current or real-world information you don't already know — release dates, sports results, news, prices, "when does X come out", "who won Y" — do not guess or make something up. Instead, respond with EXACTLY this and nothing else, no other words:

  [SEARCH: concise web search query]

  Only use this when you genuinely need up-to-date or factual information you're not confident about. Do not use it for questions about past conversations, your own personality/settings, or anything you already know.
  ` : "";

  const metaSystemInstruction = (

    `Today's date is ${new Date().toDateString()}. Use this as "today" for any date-relative reasoning or search queries — do not assume a different year from training data.\n\n` +

    `The complete contents of personality.txt are:\n\n` + `${personality}\n\n` +

    `READ ONLY MEMORY CONTEXT (not part of personality.txt):\n\n` +
    `${memory}\n\n`   +
    `CURRENT MEMORY is provided for reference.\n` +
    `Do not copy it into personality.txt.\n` +
    `Only modify memory.json when the user's requested modification target is memory.json.\n\n`+
    `The information in CURRENT MEMORY contains persistent facts and should be treated as true unless the user explicitly corrects them.\n\n` +

    modificationInstructions +
    searchInstructions

  );

  try {
    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, 90000);

    async function callOllama(prompt: string, numPredict: number): Promise<string> {
      const res = await fetch(OLLAMA_URL, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          prompt,
          stream: false,
          think: false,
          options: {
            num_predict: numPredict,
            // Left at Ollama's default (4096) this silently truncates from the
            // front of the prompt once personality + memory + recall context +
            // history + num_predict's reserved space exceed it — which drops
            // exactly the archived conversation recall depends on. Give it
            // enough headroom for that plus room to grow as archives pile up.
            num_ctx: 16384,
            temperature: 0.3
          }
        })
      });
      if (!res.ok) throw new Error("Ollama connection failed");
      const data = await res.json() as { response: string };
      return data.response;
    }

    const recentHistory = conversationHistory.slice(-20).join("\n");

    let recallContext = "";
    if (isRecallQuery) {
      if (recallDate) {
        const sessions = findArchivesByDate(recallDate)
          .map(s => ({ ...s, messages: stripStaleNoMemoryReplies(s.messages) }))
          .filter(s => s.messages.length > 0);
        const dateLabel = recallDate.toDateString();
        recallContext = sessions.length > 0
          ? `--- ARCHIVED CONVERSATION FROM ${dateLabel} ---\n` +
            sessions.map(s => s.messages.join("\n")).join("\n---\n") +
            `\n--- END ARCHIVED CONVERSATION ---\n` +
            `Every message between the markers above happened on ${dateLabel} — that is the true date of this whole block, no matter what other dates get mentioned in the conversation text itself (the user may have been asking about a different day within it). ` +
            `Answer the user's question about ${dateLabel} by summarizing what these messages show. Do not claim you have no record of ${dateLabel}: you are looking at it right now.\n\n`
          : `--- NOTE: No archived conversation was found for ${dateLabel}. Tell the user you have no record of that day. ---\n\n`;
      } else {
        const topic = extractTopicKeyword(message);
        const hits = (topic.length >= 3 ? searchArchivesByKeyword(topic) : [])
          .filter(h => !STALE_NO_MEMORY_REPLY_PATTERN.test(h.line))
          .slice(0, 30);
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
    } else if (looksLikeHeadlinesQuery) {
      console.log("FETCHING TODAY'S HEADLINES");
      const results = await fetchTodaysHeadlines();
      recallContext = results.length > 0
        ? `--- TODAY'S NEWS HEADLINES ---\n` +
          results.map((r, i) => `${i + 1}. ${r.title}\n${r.description}\n${r.url}`).join("\n\n") +
          `\n--- END TODAY'S NEWS HEADLINES ---\n` +
          `Summarize these as today's headlines for the user. ` +
          `Don't list the URLs unless asked — but if the user later asks for the link(s)/source(s), use the exact URLs shown above. Never invent a URL that isn't listed there.\n\n`
        : `--- NOTE: No current headlines could be retrieved (news search may not be configured). Tell the user you couldn't find today's news. ---\n\n`;

      // Keep these results (with real URLs) available for a follow-up like
      // "give me the link to the sites you used".
      stickySearchContext = recallContext;
      stickySearchTurnsRemaining = STICKY_RECALL_FOLLOWUP_TURNS;
    } else if (looksLikeSearchQuery) {
      // Searching with the raw message tanks result quality — conversational
      // filler ("Hey Noah, I should have granted you...") dilutes relevance
      // ranking and returns homepages instead of actual content. Have the
      // model distill a real search query first.
      const queryExtractionPrompt = `Extract a short, focused web search query (3-8 words, no quotes or extra punctuation) that would find the answer to this request. Reply with ONLY the query text and nothing else.\n\nRequest: "${message}"`;
      const rawQuery = await callOllama(queryExtractionPrompt, 30);
      const searchQuery = rawQuery.replace(/["'\n]+/g, " ").trim() || message;

      console.log("PROACTIVE WEB SEARCH TRIGGERED — extracted query:", searchQuery);
      const results = await webSearch(searchQuery);
      recallContext = results.length > 0
        ? `--- WEB SEARCH RESULTS FOR "${searchQuery}" ---\n` +
          results.map((r, i) => `${i + 1}. ${r.title}\n${r.description}\n${r.url}`).join("\n\n") +
          `\n--- END WEB SEARCH RESULTS ---\n` +
          `Answer the user's question using these results. If they don't actually answer it, say so honestly instead of guessing. ` +
          `Don't list the URLs unless asked — but if the user later asks for the link(s)/source(s), use the exact URLs shown above. Never invent a URL that isn't listed there.\n\n`
        : `--- NOTE: The web search for "${searchQuery}" returned no results (or web search isn't configured). Tell the user you couldn't find current information on this. ---\n\n`;

      // Keep these results (with real URLs) available for a follow-up like
      // "give me the link to the sites you used".
      stickySearchContext = recallContext;
      stickySearchTurnsRemaining = STICKY_RECALL_FOLLOWUP_TURNS;
    } else if (isStickySearchFollowup) {
      recallContext = stickySearchContext.replace(
        "Don't list the URLs unless asked — but if the user later asks for the link(s)/source(s), use the exact URLs shown above. Never invent a URL that isn't listed there.",
        "The user is now asking about this follow-up — use the exact URLs shown above if they're asking for links or sources. Never invent a URL that isn't listed there."
      );
      stickySearchTurnsRemaining--;

      console.log("USING STICKY SEARCH CONTEXT, turns left after this:", stickySearchTurnsRemaining);
    }

    const fullPrompt = wantsModification ? `System Instruction:\n${metaSystemInstruction}\n\n` + `User Request:\n${message}`: `System Instruction:\n${metaSystemInstruction}\n\n` + recallContext + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;
    
    console.log("WANTS MODIFICATION:", wantsModification);
    console.log("PROMPT SENT TO MODEL:");
    console.log(fullPrompt);

    console.log(`[MODEL] ${selectedModel} | complex=${isComplex}`);

    let aiResponse = await callOllama(fullPrompt, isComplex ? 3000 : 80);

    console.log("RAW AI RESPONSE:");
    console.log(aiResponse);

    const searchMatch = aiResponse.match(SEARCH_TAG_PATTERN);
    if (searchMatch) {
      const searchQuery = searchMatch[1].trim();
      console.log("WEB SEARCH REQUESTED:", searchQuery);

      const results = await webSearch(searchQuery);
      const searchContext = results.length > 0
        ? `--- WEB SEARCH RESULTS FOR "${searchQuery}" ---\n` +
          results.map((r, i) => `${i + 1}. ${r.title}\n${r.description}\n${r.url}`).join("\n\n") +
          `\n--- END WEB SEARCH RESULTS ---\n` +
          `Answer the user's original question using these results. If they don't actually answer it, say so honestly instead of guessing. ` +
          `Don't list the URLs unless asked — but if the user later asks for the link(s)/source(s), use the exact URLs shown above. Never invent a URL that isn't listed there.\n\n`
        : `--- NOTE: The web search for "${searchQuery}" returned no results (or web search isn't configured). Tell the user you couldn't find current information on this. ---\n\n`;

      // Keep these results (with real URLs) available for a follow-up like
      // "give me the link to the sites you used".
      stickySearchContext = searchContext;
      stickySearchTurnsRemaining = STICKY_RECALL_FOLLOWUP_TURNS;

      const followUpPrompt = `System Instruction:\n${metaSystemInstruction}\n\n` + searchContext + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;

      aiResponse = await callOllama(followUpPrompt, isComplex ? 3000 : 400);

      console.log("FINAL AI RESPONSE AFTER SEARCH:");
      console.log(aiResponse);
    }

    clearTimeout(timeout);

    console.log("RESPONSE LENGTH:", aiResponse?.length ?? 0);

    let jsonModifications: Modification[] = [];
    let isPureJsonResponse = false;
    let matchedJsonBlock = "";

    try {
      const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        isPureJsonResponse = true;
        matchedJsonBlock = jsonMatch[0];

        pendingCommit = parsed.commit ?? null;

        if (Array.isArray(parsed.modifications)) {
            jsonModifications =
                parsed.modifications as Modification[];

            console.log(
                "JSON MODIFICATIONS FOUND:",
                jsonModifications.length
            );
        }
      }
    } catch {
      // Not JSON, continue normally
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

    const hasAnyProposal = updates.length > 0 || jsonModifications.length > 0;

    // Noah issometimes asked to modify memory/personality but just chats back without actually proposing any changes
    // Catches that here instead of forwarding the false confirmation.
    const silentModificationFailure = wantsModification && !hasAnyProposal;

    if (silentModificationFailure) {
      console.warn("MODIFICATION REQUEST PRODUCED NO PROPOSAL — model replied without drafting a change:", aiResponse);
    }

    if (!aiResponse.includes("I cannot execute") && !aiResponse.includes("I am an AI model")) {

      const isRefusal = aiResponse.includes("cannot execute") || aiResponse.includes("cannot modify") || aiResponse.includes("I am an AI model");

      if (!isRefusal) {
        const historySafeResponse = silentModificationFailure
          ? "[Modification request failed: no change was drafted]"
          : isPureJsonResponse
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

    if (silentModificationFailure) {
      return res.json({
        response: "That didn't actually save — I wasn't able to draft the change, so nothing was updated. Try again, or rephrase it (e.g. \"remember that ...\").",
        hasProposedChanges: false
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

    const cleanText = isPureJsonResponse
      ? aiResponse.replace(matchedJsonBlock, "").trim()
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

// VS Code's Git extension reads .git/COMMIT_EDITMSG and shows it in the Source Control input box whenever that box is empty. 
// Git already writes this file as part of `git commit -m`, but we write it explicitly here too
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