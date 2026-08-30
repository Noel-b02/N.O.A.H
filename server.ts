import express from 'express';
import * as fs from 'fs';
import { execFileSync, spawn, ChildProcess } from 'child_process';
import path from 'path';
import dotenv from 'dotenv';
import { randomUUID } from 'crypto';
import { startTelegramBot } from './telegram';

dotenv.config();

const app = express();
const PORT = 3000;

// Default express.json() body limit is 100kb — far too small for a
// base64-encoded photo attachment (which also inflates ~33% over the raw
// file size), so this needs raising explicitly.
app.use(express.json({ limit: "15mb" }));
app.use(express.static(path.join(__dirname, 'public')));

const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434/api/generate";
const SPEECH_SERVICE_URL = process.env.SPEECH_SERVICE_URL ?? "http://localhost:5001";
const VISION_SERVICE_URL = process.env.VISION_SERVICE_URL ?? "http://localhost:5002";

const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL ?? "qwen3.5:4b";
const CODE_MODEL = process.env.OLLAMA_CODE_MODEL ?? "qwen3.5:9b";
// Only needed for document Q&A (see extractDocumentText/chunkText below) —
// chat itself works fine without this model ever being pulled.
const EMBEDDING_MODEL = process.env.OLLAMA_EMBEDDING_MODEL ?? "nomic-embed-text";
const EMBEDDINGS_URL = OLLAMA_URL.replace(/\/api\/generate$/, "/api/embeddings");

const SEARXNG_URL = process.env.SEARXNG_URL ?? "http://localhost:8080";
const FUSION_BRIDGE_URL = process.env.FUSION_BRIDGE_URL ?? "http://localhost:9000";

// Bambu Studio CLI (slicing) — see PRINTER_SETUP.md. Profile paths are left
// blank rather than guessed: where Bambu Studio stores its bundled system
// profiles varies and hasn't been confirmed against a real install yet.
const BAMBU_STUDIO_EXE = process.env.BAMBU_STUDIO_EXE ?? "C:\\Program Files\\Bambu Studio\\bambu-studio.exe";
const BAMBU_MACHINE_PROFILE = process.env.BAMBU_MACHINE_PROFILE ?? "";
const BAMBU_PROCESS_PROFILE = process.env.BAMBU_PROCESS_PROFILE ?? "";
const BAMBU_FILAMENT_PROFILE = process.env.BAMBU_FILAMENT_PROFILE ?? "";

const IMAGE23D_DIR = path.join(__dirname, "image23d");
const IMAGE23D_TMP_DIR = path.join(IMAGE23D_DIR, "tmp");
// Needs ~6GB VRAM in shape-only mode, so it runs exclusively (Ollama
// unloaded, speech service stopped) rather than alongside them.
const HUNYUAN3D_DIR = path.join(IMAGE23D_DIR, "Hunyuan3D-2");
const HUNYUAN3D_PYTHON = path.join(HUNYUAN3D_DIR, "venv", "Scripts", "python.exe");

// Lives under public/ so express.static() serves the .glb files directly —
// no separate file-serving endpoint needed. Kept distinct from
// IMAGE23D_TMP_DIR: that one gets wiped per-request, these are meant to
// stick around so the "view model" tab in the UI keeps working after the
// chat response that created it has scrolled by.
const MODELS_DIR = path.join(__dirname, "public", "generated-models");
// Leading dot deliberately — express.static() ignores dotfiles by default,
// so this internal bookkeeping (unlike the .glb files themselves) isn't
// directly fetchable over HTTP.
const MODELS_INDEX_FILE = path.join(MODELS_DIR, ".index.json");
// Cap on UNSAVED models specifically — anything explicitly saved via the
// UI is exempt, so this only prunes the rolling "recent, not saved" set.
const MAX_UNSAVED_MODELS = 20;

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

// Keeps the last retrieved archive context available for a few follow-up
// turns (e.g. "give me specifics", "quote it") so the model doesn't have to
// improvise from its own prior summary once the original recall query has
// scrolled out of isRecallQuery detection.
// Declared before archiveSession() is called below — that call reaches into
// these on startup whenever there's leftover history to archive, so they
// must already be initialized by the time it runs, not just by the time
// their own line is reached.
let stickyRecallContext = "";
let stickyRecallTurnsRemaining = 0;
const STICKY_RECALL_FOLLOWUP_TURNS = 3;

// Same idea, for web search: a follow-up like "give me the link to the sites
// you used" needs the same search results (with real URLs) that answered the
// original question, not just its own text summary from conversation history.
let stickySearchContext = "";
let stickySearchTurnsRemaining = 0;

let conversationHistory: string[] = loadHistory();
archiveSession("startup", previousSessionEndedAt);

let requestInFlight = false;
let pendingCommit: CommitInfo | null = null;
// Surfaced on the HUD as MESH_GEN — withGpuExclusive() is the only thing
// that sets this, since that's the only class of work slow/disruptive
// enough (unloads Ollama/speech, can run for minutes) to be worth telling
// the user about explicitly rather than just reading GPU utilization.
let gpuExclusiveTaskRunning = false;

interface PendingPrintJob {
  gcodePath: string;
  subject: string;
  estimatedTimeMin: number | null;
  estimatedFilamentG: number | null;
  createdAt: number;
}
// Deliberately NOT decremented/cleared by an unrelated intervening message
// like the sticky-recall/search followups above are — a pending print
// should survive small talk and only clear on explicit confirm, explicit
// cancel, or PENDING_PRINT_EXPIRY_MS, since "reply print to start it"
// implies the user can reply whenever, not necessarily next turn.
let pendingPrintJob: PendingPrintJob | null = null;
const PENDING_PRINT_EXPIRY_MS = 10 * 60 * 1000;


function loadHistory(): string[] {
  try {
    return JSON.parse(
      fs.readFileSync(HISTORY_FILE, "utf8")
    );
  } catch {
    return [];
  }
}

// Creates memories/history/ on demand — needed for a fresh clone, where
// nothing under memories/ is tracked in git (it's all personal data) and so
// the directory itself doesn't exist until something is actually saved.
function saveHistory(): void {
  fs.mkdirSync(path.dirname(HISTORY_FILE), { recursive: true });
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
  pendingPrintJob = null;

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

function extractXmlTag(block: string, tag: string): string {
  const match = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  if (!match) return "";
  const cdataMatch = match[1].match(/^<!\[CDATA\[([\s\S]*?)\]\]>$/);
  const raw = cdataMatch ? cdataMatch[1] : match[1];
  return raw
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#0?39;/g, "'")
    .replace(/<\/?[^>]+>/g, "")
    .trim();
}

async function fetchRssHeadlines(feedUrl: string, limit = 5): Promise<WebSearchResult[]> {
  try {
    const res = await fetch(feedUrl);
    if (!res.ok) {
      console.error("RSS fetch failed:", feedUrl, res.status);
      return [];
    }

    const xml = await res.text();
    const items: WebSearchResult[] = [];
    const itemPattern = /<item>([\s\S]*?)<\/item>/g;
    let match;

    while ((match = itemPattern.exec(xml)) !== null && items.length < limit) {
      const block = match[1];
      const title = extractXmlTag(block, "title");
      const url = extractXmlTag(block, "link");
      const description = extractXmlTag(block, "description");
      if (title) items.push({ title, description, url });
    }

    return items;
  } catch (err) {
    console.error("RSS fetch error:", feedUrl, err);
    return [];
  }
}

// "Give me today's headlines" is a fundamentally different task from a
// factual lookup like "who won the world cup" — it's not searching FOR
// anything specific, so a plain web search just surfaces static homepages
// (which rank highest for a bare term like "headlines"). SearXNG's news
// category plus a day-range filter routes through actual news-aggregator
// engines and returns real, dated, current articles instead.
//
// In practice, of SearXNG's ~17 configured news engines, only "reuters" is
// consistently working right now — the rest are either broken by upstream
// site changes (e.g. bing_news.py has an unfixed parser bug) or actively
// blocked by CAPTCHA/rate-limiting (startpage, duckduckgo, brave), which is
// a permanent cat-and-mouse reality of scraping-based search, not something
// that stays fixed once patched. BBC's own RSS feed is added directly as a
// second reliable, real source — no scraping, no CAPTCHA risk, no API key —
// so headlines aren't reliant on a single outlet. Skipped for topic-specific
// searches, since BBC's general front page wouldn't be relevant to those.
async function fetchTodaysHeadlines(topic?: string): Promise<WebSearchResult[]> {
  const searxResults = await webSearch(topic?.trim() || "news", { categories: "news", timeRange: "day" });
  if (topic?.trim()) return searxResults;

  const rssResults = await fetchRssHeadlines("http://feeds.bbci.co.uk/news/rss.xml", 5);
  return [...searxResults, ...rssResults];
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
// The optional " (Telegram)" tag between "Assistant" and ":" (see channelTag
// in /api/chat) must still match here, or every stale reply sent over
// Telegram would leak into recall context untouched.
const STALE_NO_MEMORY_REPLY_PATTERN = /^Assistant(?: \(Telegram\))?:.*\b(don't have access to (our|any|the)|no record of|not in my (immediate )?memory|memory (starts fresh|is limited to|only (goes|holds))|no archived (conversation|mentions))\b/i;

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

// "I can't remember X" is the user describing their OWN forgetfulness — not
// an instruction for Noah to store anything — but it contains "remember"
// just as literally as an imperative would, so it needs the same carve-out.
const FORGETFUL_STATEMENT_PATTERN = /\b(i|we)\s+(can'?t|cannot|couldn'?t|could not|don'?t|do not)\s+remember\b/i;

function extractTopicKeyword(message: string): string {
  const aboutMatch = message.match(/about\s+(.+?)[\?\.!]?$/i);
  if (aboutMatch) return aboutMatch[1].trim();
  return message.replace(RECALL_TRIGGER_PATTERN, "").trim();
}

function loadFile(filepath: string): string {
  try { return fs.readFileSync(filepath, 'utf8'); } catch { return ""; }
}

function writeFile(filepath: string, content: string): void {
  // Same reasoning as saveHistory() — memories/memory.json's parent
  // directory isn't guaranteed to exist on a fresh clone.
  fs.mkdirSync(path.dirname(filepath), { recursive: true });
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

function getVenvPythonPath(baseDir: string): string {
  const isWindows = process.platform === "win32";
  return path.join(
    baseDir,
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

  const pythonPath = getVenvPythonPath(SPEECH_SERVICE_DIR);

  if (!fs.existsSync(pythonPath)) {
    console.warn(
      `[speech] No venv Python found at ${pythonPath} — voice features will be unavailable ` +
      `until the speech service venv is set up (see speech/requirements.txt) or started manually.`
    );
    return;
  }

  console.log(`[speech] Starting speech service (${pythonPath})...`);

  // detached: true (Windows: CREATE_NEW_PROCESS_GROUP) is the same
  // convention every other background process launch in this file already
  // uses (ollama serve, Docker Desktop, Fusion 360, Bambu Connect) — this
  // was the one spawn that didn't have it, and the one spawn that shares
  // the console via stdio: "inherit" rather than "ignore" (needed here so
  // the speech service's logs show up inline, unlike those fire-and-forget
  // launches). Without its own process group, a child sharing the parent's
  // console on Windows sits in the same Ctrl+C signal group as the parent
  // — and uvicorn/asyncio's own startup on Windows is a known trigger for
  // a spurious console Ctrl+C event, which is exactly what was killing the
  // whole `tsx watch server.ts` process right after this log line.
  speechServiceProcess = spawn(
    pythonPath,
    ["-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", SPEECH_SERVICE_PORT],
    { cwd: SPEECH_SERVICE_DIR, stdio: "inherit", detached: true }
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

// --- Vision service (camera + facial recognition) auto-start ---
// CPU-only end to end (insightface/onnxruntime), so unlike speech there's no
// VRAM contention to worry about — this never needs withGpuExclusive().
// Defaults to NOT auto-starting, inverted from speech's default-on: a
// continuously-active webcam is a materially more visible trust surface than
// an on-demand mic, and this repo is public — nobody who clones it should
// get a webcam silently activated by an unrelated git pull + restart. See
// VISION_SETUP.md.
const VISION_SERVICE_DIR = process.env.VISION_SERVICE_DIR ?? path.join(process.cwd(), "vision");
const VISION_SERVICE_PORT = process.env.VISION_SERVICE_PORT ?? "5002";
const VISION_SERVICE_AUTOSTART = (process.env.VISION_SERVICE_AUTOSTART ?? "false").toLowerCase() === "true";

let visionServiceProcess: ChildProcess | null = null;

async function isVisionServiceRunning(): Promise<boolean> {
  try {
    const res = await fetch(`${VISION_SERVICE_URL}/health`, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}

async function startVisionService(): Promise<void> {
  if (!VISION_SERVICE_AUTOSTART) {
    console.log("[vision] Auto-start disabled (default — set VISION_SERVICE_AUTOSTART=true to enable).");
    return;
  }

  if (await isVisionServiceRunning()) {
    console.log("[vision] Vision service already running on its own — skipping auto-start.");
    return;
  }

  const pythonPath = getVenvPythonPath(VISION_SERVICE_DIR);

  if (!fs.existsSync(pythonPath)) {
    console.warn(
      `[vision] No venv Python found at ${pythonPath} — facial recognition will be unavailable ` +
      `until the vision service venv is set up (see vision/requirements.txt) or started manually.`
    );
    return;
  }

  console.log(`[vision] Starting vision service (${pythonPath})...`);

  // detached: true from the start — a missing detached: true on the speech
  // service's original spawn caused a spurious console Ctrl+C to kill the
  // whole tsx watch process group on Windows (see startSpeechService above).
  visionServiceProcess = spawn(
    pythonPath,
    ["-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", VISION_SERVICE_PORT],
    { cwd: VISION_SERVICE_DIR, stdio: "inherit", detached: true }
  );

  visionServiceProcess.on("error", (err) => {
    console.warn(`[vision] Failed to start vision service: ${err.message}`);
    visionServiceProcess = null;
  });

  visionServiceProcess.on("exit", (code, signal) => {
    console.log(`[vision] Vision service process exited (code=${code}, signal=${signal})`);
    visionServiceProcess = null;
  });
}

function stopVisionService(): void {
  if (visionServiceProcess && !visionServiceProcess.killed) {
    console.log("[vision] Stopping vision service...");
    visionServiceProcess.kill();
  }
}

// Best-effort — if a model isn't currently loaded this just errors quietly,
// which is fine, that's already the state we want.
function unloadOllamaModels(): void {
  for (const model of [CHAT_MODEL, CODE_MODEL]) {
    try {
      execFileSync("ollama", ["stop", model], { stdio: "pipe", timeout: 10000 });
    } catch {
      // Not loaded, or ollama isn't on PATH — either way, nothing to clean up.
    }
  }
}

// Some local models (Hunyuan3D's shape stage, ~6GB) need more VRAM than an
// 8GB card has free once Ollama and the speech service are also holding
// their share — there's no way to fit all three at once on this hardware.
// This frees the other two first, runs the heavy step, then always restores
// them afterward (even on failure) via finally. Voice input/output is
// unavailable for the duration; the next Ollama reply after this also pays
// a one-time model-reload cost. Once there's enough VRAM for everything at
// once, this wrapper stops being necessary — it can just be removed instead
// of updated.
async function withGpuExclusive<T>(fn: () => Promise<T>): Promise<T> {
  unloadOllamaModels();
  stopSpeechService();
  gpuExclusiveTaskRunning = true;
  // kill() is fire-and-forget (SIGTERM) — give the process a moment to
  // actually exit and release VRAM before starting the heavy step.
  await new Promise(r => setTimeout(r, 2000));
  try {
    return await fn();
  } finally {
    startSpeechService();
    gpuExclusiveTaskRunning = false;
  }
}

startSpeechService();
startVisionService();
startTelegramBot();

// --- Bambu Connect (official print handoff) ---
// See PRINTER_SETUP.md. Bambu Lab's own desktop relay app — chosen over a
// direct MQTT/FTP connection specifically because that path requires
// LAN-only + Developer Mode on the printer, which drops Bambu Handy and
// remote cloud access entirely. Bambu Connect keeps the printer on normal
// Cloud mode. Trade-off: this is a one-directional handoff (Noah opens the
// file in Bambu Connect, the user confirms in ITS window) — there's no
// channel back to Noah for live print status, so unlike a direct MQTT link
// this can't report PRINTING/PAUSED/percent-complete.
function isBambuConnectInstalled(): boolean {
  try {
    execFileSync("reg", ["query", "HKCR\\bambu-connect"], { stdio: "pipe", timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}

// URL scheme is beta and undocumented beyond Bambu's own wiki page — Bambu
// Lab could change it without notice. Confirmed live: launching via
// `cmd /c start` mangles the URL — cmd.exe reinterprets '&' as a command
// separator and '%XX' percent-encoding as environment-variable expansion,
// so Bambu Connect ended up receiving an empty path. rundll32's
// FileProtocolHandler hands the URL straight to Windows' registered
// protocol handler without going through cmd.exe's parsing at all, which
// fixed it (verified: Bambu Connect showed the correct filename after
// switching to this).
function launchBambuConnect(gcodePath: string, name: string): { success: boolean; error?: string } {
  const url = `bambu-connect://import-file?path=${encodeURIComponent(gcodePath)}&name=${encodeURIComponent(name)}&version=1.0.0`;
  try {
    spawn("rundll32", ["url.dll,FileProtocolHandler", url], { detached: true, stdio: "ignore" }).unref();
    return { success: true };
  } catch (err) {
    return { success: false, error: (err as Error).message };
  }
}

app.get('/api/status', (_, res) => {
  res.json({
    activeDraftBranch,
    pendingFiles,
    historySize: conversationHistory.length
  });
});

interface GpuStats {
  utilizationPercent: number;
  temperatureC: number;
  vramUsedMB: number;
  vramTotalMB: number;
}

function getGpuStats(): GpuStats | null {
  try {
    const output = execFileSync("nvidia-smi", [
      "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
      "--format=csv,noheader,nounits"
    ], { encoding: "utf8", timeout: 3000 });

    const [util, temp, used, total] = output.trim().split(",").map(s => parseFloat(s.trim()));
    if ([util, temp, used, total].some(Number.isNaN)) return null;

    return { utilizationPercent: util, temperatureC: temp, vramUsedMB: used, vramTotalMB: total };
  } catch {
    // No NVIDIA GPU, driver not installed, or nvidia-smi not on PATH.
    return null;
  }
}

// Real, live data (actual file sizes of models already pulled) — same
// base-URL-rewrite trick already used for /api/version and /api/embeddings
// above, applied to Ollama's own /api/tags endpoint. Not used anywhere in
// this project before now.
const OLLAMA_TAGS_URL = OLLAMA_URL.replace(/\/api\/generate$/, "/api/tags");

interface PulledModel {
  name: string;
  sizeBytes: number;
}

async function listPulledModels(): Promise<PulledModel[]> {
  try {
    const res = await fetch(OLLAMA_TAGS_URL, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return [];
    const data = await res.json() as { models?: { name: string; size: number }[] };
    return (data.models ?? []).map(m => ({ name: m.name, sizeBytes: m.size }));
  } catch {
    return [];
  }
}

// Approximate VRAM figures for common Q4_K_M-quantization builds of models
// a home user would realistically consider pulling — manually curated, NOT
// live-confirmed the way the image-pipeline VRAM figures elsewhere in this
// file are (those come from real measured peaks; these are reasonable
// estimates for models that may not even be pulled yet). Needs occasional
// manual upkeep as new model families become popular — an accepted,
// deliberate limitation, not an oversight. The reply text built from this
// table says so explicitly, not just this comment.
const OLLAMA_MODEL_VRAM_TABLE: { name: string; approxVramGB: number }[] = [
  { name: "llama3.2:1b", approxVramGB: 1.5 },
  { name: "llama3.2:3b", approxVramGB: 2.5 },
  { name: "llama3.1:8b", approxVramGB: 5.5 },
  { name: "llama3.1:70b", approxVramGB: 43 },
  { name: "qwen2.5:7b", approxVramGB: 5 },
  { name: "qwen2.5:14b", approxVramGB: 9.5 },
  { name: "qwen2.5:32b", approxVramGB: 20 },
  { name: "mistral:7b", approxVramGB: 5 },
  { name: "mixtral:8x7b", approxVramGB: 27 },
  { name: "phi3:mini", approxVramGB: 2.5 },
  { name: "phi3:medium", approxVramGB: 8.5 },
  { name: "gemma2:9b", approxVramGB: 6.5 },
  { name: "gemma2:27b", approxVramGB: 17 },
  { name: "codellama:13b", approxVramGB: 8.5 },
  { name: "deepseek-r1:7b", approxVramGB: 5 },
  { name: "deepseek-r1:14b", approxVramGB: 9.5 }
];

interface ModelRecommendationData {
  totalVramGB: number | null;
  pulled: { name: string; sizeGB: number; fits: boolean | null }[];
  suggestions: { name: string; approxVramGB: number }[];
}

// Pure data-gathering, shared by the chat-based model-recommend intent
// (which wraps this in recallContext prose) and GET /api/models/overview
// (which returns it as-is plus an LLM-generated pros/cons layer) — kept as
// one function so the two surfaces can't drift on the fits/exceeds-VRAM
// logic the way a duplicated copy eventually would.
function buildModelRecommendationData(pulled: PulledModel[], gpu: GpuStats | null): ModelRecommendationData {
  const pulledNames = new Set(pulled.map(m => m.name));
  const totalVramGB = gpu ? gpu.vramTotalMB / 1024 : null;

  const pulledData = pulled.map(m => {
    const sizeGB = m.sizeBytes / 1e9;
    const fits = totalVramGB !== null ? sizeGB <= totalVramGB : null;
    return { name: m.name, sizeGB, fits };
  });

  // Only suggest table entries not already pulled, and only ones that
  // would plausibly fit — "plausibly" because these are approximate
  // figures, not measured peaks, so the fit call itself is a judgment
  // call, not a guarantee.
  const suggestions = totalVramGB !== null
    ? OLLAMA_MODEL_VRAM_TABLE.filter(m => !pulledNames.has(m.name) && m.approxVramGB <= totalVramGB * 0.85)
    : [];

  return { totalVramGB, pulled: pulledData, suggestions };
}

// Standalone Ollama call usable outside the /api/chat request cycle — the
// cascade's own callOllama() is a closure nested inside that handler
// (captures a request-scoped AbortController/selectedModel), so it can't be
// called from an independent REST route. Used by runModelComparison below
// and GET /api/models/overview's pros/cons generation.
async function callOllamaModel(prompt: string, numPredict: number, model: string, timeoutMs = 60000): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(OLLAMA_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        // Without this, some models (confirmed live: qwen3.5:4b) default to
        // "thinking" mode and put their real answer in a separate
        // `thinking` field, leaving `response` empty until num_predict runs
        // out mid-thought — matching the nested callOllama() in /api/chat,
        // which passes this explicitly for the same reason.
        think: false,
        options: { num_predict: numPredict, num_ctx: 16384, temperature: 0.3 }
      })
    });
    if (!res.ok) throw new Error(`Ollama connection failed: HTTP ${res.status} ${await res.text().catch(() => "")}`.trim());
    const data = await res.json() as { response: string };
    return data.response;
  } finally {
    clearTimeout(timeout);
  }
}

// Shared by the chat-based "compare X and Y on: ..." intent and
// POST /api/models/compare — runs the same prompt against two models in
// parallel, each independently try/caught so one failure doesn't blank out
// the other's real result.
async function runModelComparison(modelA: string, modelB: string, prompt: string): Promise<{ model: string; response: string | null; error?: string }[]> {
  const runOne = async (modelName: string): Promise<{ model: string; response: string | null; error?: string }> => {
    try {
      // 400 (the original limit) cut real answers off mid-sentence for
      // anything beyond a short conversational reply — confirmed live with
      // a "write a function, with comments" prompt, where both models were
      // still mid-explanation when they hit it. Comparison is already a
      // deliberate, wait-for-it user action, so a generous budget (and a
      // matching longer timeout below) is worth it over truncated answers.
      const raw = await callOllamaModel(prompt, 1500, modelName, 120000);
      return { model: modelName, response: raw.trim() };
    } catch (err: any) {
      return { model: modelName, response: null, error: err.message };
    }
  };
  return Promise.all([runOne(modelA), runOne(modelB)]);
}

async function isServiceHealthy(url: string, timeoutMs = 2000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}

// Fusion 360 auto-updates itself, and each update gets a new hash-named
// folder under webdeploy/production — hardcoding a specific one would break
// the next time it updates. Globbing for whichever folder currently has the
// launcher keeps this working across updates with no maintenance.
function findFusionLauncher(): string | null {
  const base = path.join(process.env.LOCALAPPDATA ?? "", "Autodesk", "webdeploy", "production");
  try {
    for (const dir of fs.readdirSync(base)) {
      const launcherPath = path.join(base, dir, "FusionLauncher.exe");
      if (fs.existsSync(launcherPath)) return launcherPath;
    }
  } catch {
    // webdeploy folder missing entirely — Fusion likely isn't installed
    // under the default per-user location.
  }
  return null;
}

// Launches Fusion 360 if the bridge isn't reachable, then polls (rather than
// guessing a fixed delay) since it's a heavy desktop app that can take a
// while to start — real feedback beats a blind wait. Capped at 60s so the
// rest of the pipeline (search/generation) still fits inside the overall
// 180s request timeout on a first-launch request.
async function ensureFusionBridgeAvailable(): Promise<{ available: boolean; justLaunched: boolean }> {
  if (await isServiceHealthy(`${FUSION_BRIDGE_URL}/health`)) {
    return { available: true, justLaunched: false };
  }

  const launcherPath = findFusionLauncher();
  if (!launcherPath) {
    return { available: false, justLaunched: false };
  }

  spawn(launcherPath, [], { detached: true, stdio: "ignore" }).unref();

  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 5000));
    if (await isServiceHealthy(`${FUSION_BRIDGE_URL}/health`)) {
      return { available: true, justLaunched: true };
    }
  }
  return { available: false, justLaunched: true };
}

interface FusionExecutionResult {
  success: boolean;
  error?: string | null;
}

// Talks to the NoahFusionBridge add-in (fusion-bridge/NoahFusionBridge) —
// a local HTTP server that only exists while Fusion 360 is running with that
// add-in active. Executes generated Python against the live Fusion API on
// Fusion's main thread (see the add-in's own comments for why that hand-off
// is necessary) and reports back whether it actually succeeded.
async function executeFusionScript(code: string): Promise<FusionExecutionResult> {
  try {
    const res = await fetch(`${FUSION_BRIDGE_URL}/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
      signal: AbortSignal.timeout(35000)
    });

    if (!res.ok) {
      return { success: false, error: `Fusion bridge returned HTTP ${res.status}` };
    }

    return await res.json() as FusionExecutionResult;
  } catch (err: any) {
    return {
      success: false,
      error: `Could not reach the Fusion 360 bridge at ${FUSION_BRIDGE_URL}: ${err.message}. ` +
        `Is Fusion 360 running with the NoahFusionBridge add-in active?`
    };
  }
}

// Requires an explicit "3d model"/"cad model"/"in fusion" mention alongside
// a creation verb — kept deliberately narrow since this triggers real code
// execution inside a running Fusion 360 session, not just a chat response.
// Top-level (moved out of the handler) so isStickySearchFollowup/
// isStickyRecallFollowup can test against it directly — confirmed live on
// master: a sticky search/headlines follow-up window swallowed an
// unrelated, clearly-distinct request before it ever reached its own
// trigger pattern, since those sticky flags are computed before
// looksLikeFusionRequest exists as a variable and couldn't previously
// exclude it.
const FUSION_TRIGGER_PATTERN = /\b(create|make|generate|build|design|remove|delete|clear)\b.{0,40}\b(3d models?|3d shapes?|cad models?|in fusion(?: ?360)?)\b/i;

// Lets a request override the auto-searched reference image with a specific
// one (e.g. "...in fusion 360: https://example.com/photo.jpg") — needed for
// subjects like "controller holder" where every real product photo shows the
// item in use, so no search phrasing can reliably find a clean shot.
const IMAGE_URL_PATTERN = /(https?:\/\/\S+?\.(?:jpe?g|png|webp|gif)(?:\?\S*)?)(?=[\s.,;!?]|$)/i;

function extractImageUrl(message: string): string | null {
  const match = message.match(IMAGE_URL_PATTERN);
  return match ? match[1] : null;
}

// Pulls the named subject out of an image-based Fusion request (e.g. "create
// a 3d model of spiderman in fusion" -> "spiderman") so it can be used as
// both the image search query and the reply text, without needing a
// dedicated Ollama call to extract a noun phrase.
// Matches a trailing sizing instruction (e.g. ", make it 5cm", "5cm tall",
// "about 2 inches") so it can be stripped before subject extraction — left
// in place, it either gets glued onto the subject text (breaking the image
// search query) or its own "make" verb gets caught by the verb-stripping
// regex below, leaving a mangled leftover like "it 5cm".
const SIZE_CLAUSE_PATTERN = /,?\s*\b(?:make (?:it|the model)\s+|about\s+|roughly\s+|sized?\s+(?:at\s+)?|at\s+)?\d+(?:\.\d+)?\s?(?:millimeters?|mm|centimeters?|cm|meters?|m|inches?|in|feet|ft)\b(?:\s+(?:tall|high|wide|long|in\s+(?:height|size)))?/i;

function extractFusionSubject(message: string): string {
  const withoutUrl = message.replace(IMAGE_URL_PATTERN, "").trim();
  const withoutSize = withoutUrl.replace(SIZE_CLAUSE_PATTERN, "").trim();
  const ofMatch = withoutSize.match(/\bof\s+(?:a|an|the)\s+(.+?)[\?\.!]*$/i) ?? withoutSize.match(/\bof\s+(.+?)[\?\.!]*$/i);
  const raw = ofMatch ? ofMatch[1] : withoutSize.replace(/\b(create|make|generate|build|design)\b/gi, "").replace(/\b(a|an|the)\b/gi, "");
  // The "of ..." match is greedy to end-of-string, so it swallows trailing
  // trigger phrases too (e.g. "of a fox in fusion 360" -> "fox in fusion
  // 360") — strip those out regardless of which branch produced the string.
  const cleaned = raw
    .replace(/\b(3d models?|3d shapes?|cad models?|in fusion(?: ?360)?)\b/gi, "")
    .replace(/[:,.\s]+$/, "")
    .trim();
  // A supplied URL leaves only a filler word behind (e.g. "this", "that") —
  // swap in something readable for the reply text instead.
  return cleaned && !/^(this|that|it)$/i.test(cleaned) ? cleaned : "the provided reference image";
}

// Extracts plain text from an uploaded document. PDF/DOCX libraries are
// require()'d lazily inside here rather than imported at module load,
// simply so this file doesn't pay their load cost unless a document is
// actually attached.
async function extractDocumentText(base64: string, mimeType: string, fileName: string): Promise<string> {
  const buffer = Buffer.from(base64, "base64");
  const ext = path.extname(fileName).toLowerCase();

  if (mimeType === "application/pdf" || ext === ".pdf") {
    // pdf-parse v2's API is class-based (new PDFParse({data}).getText()),
    // not the plain callable-function API older v1 examples show —
    // confirmed directly against the installed version (2.4.5).
    const { PDFParse } = require("pdf-parse");
    const parser = new PDFParse({ data: buffer });
    const result = await parser.getText();
    await parser.destroy();
    return result.text;
  }
  if (ext === ".docx" || mimeType.includes("officedocument.wordprocessingml")) {
    const mammoth = require("mammoth");
    return (await mammoth.extractRawText({ buffer })).value;
  }
  return buffer.toString("utf8"); // .txt / .md / anything else
}

// Rejects absurdly large uploads outright rather than silently generating
// hundreds of embedding calls for something that was probably attached by
// mistake — roughly the length of a long novel.
const MAX_DOCUMENT_CHARS = 600000;

// Simple paragraph-aware greedy packing, not a tokenizer — good enough for
// chunk boundaries that roughly track topic shifts, without pulling in a
// tokenizing dependency for it. CHUNK_OVERLAP carries a little of the
// previous chunk forward so a fact split across a chunk boundary isn't
// lost to retrieval entirely.
const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 150;

function chunkText(text: string): string[] {
  const paragraphs = text.replace(/\r\n/g, "\n").split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  for (const para of paragraphs) {
    if (para.length > CHUNK_SIZE) {
      if (current) { chunks.push(current); current = ""; }
      for (let i = 0; i < para.length; i += CHUNK_SIZE - CHUNK_OVERLAP) {
        chunks.push(para.slice(i, i + CHUNK_SIZE));
      }
      continue;
    }
    if (current && (current.length + para.length + 2) > CHUNK_SIZE) {
      chunks.push(current);
      current = current.slice(-CHUNK_OVERLAP) + "\n\n" + para;
    } else {
      current = current ? current + "\n\n" + para : para;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

// Mirrors callOllama's shape but hits Ollama's separate /api/embeddings
// endpoint (not /api/generate) with the embedding model rather than a chat
// model.
async function getEmbedding(text: string): Promise<number[]> {
  const res = await fetch(EMBEDDINGS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBEDDING_MODEL, prompt: text })
  });
  if (!res.ok) {
    throw new Error(`Embedding request failed: ${res.status} ${await res.text().catch(() => "")}`);
  }
  const data = await res.json();
  return data.embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

interface DocumentEntry {
  id: string;
  fileName: string;
  chunks: { text: string; embedding: number[] }[];
  createdAt: string;
}

// In-memory only, deliberately not written to disk — the source file the
// user attached still exists untouched on their machine, so re-attaching
// after a restart is cheap. Unlike conversationHistory (irreplaceable),
// there's no real data-loss case to protect against here.
const documentStore = new Map<string, DocumentEntry>();
// Soft cap on how many documents stay active in memory at once — same
// "prune oldest" idea as MAX_UNSAVED_MODELS, just against a Map instead of
// a JSON-backed index.
const MAX_ACTIVE_DOCUMENTS = 5;

// Brute-force cosine similarity over a document's chunks — this is a
// single-user local app with at most a few hundred chunks per document, so
// a real vector database would be pure overhead for no measurable benefit.
async function retrieveRelevantChunks(documentId: string, query: string, topK: number = 5): Promise<string[]> {
  const doc = documentStore.get(documentId);
  if (!doc) return [];
  const queryEmbedding = await getEmbedding(query);
  return doc.chunks
    .map(c => ({ text: c.text, score: cosineSimilarity(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(c => c.text);
}

// Looks for a "<number> <unit>" size mention (e.g. "5cm", "2 inches", "10mm
// tall") and converts it to centimeters — the unit the Fusion import step
// already treats the generated mesh's raw output numbers as. Requires the unit to
// immediately follow a digit, so ordinary words like "fox in fusion 360"
// (which contains "in") don't false-positive.
const SIZE_PATTERN = /(\d+(?:\.\d+)?)\s?(millimeters?|mm|centimeters?|cm|meters?|m\b|inches?|in\b|feet|ft\b)/i;

function extractTargetSizeCm(message: string): number | null {
  const match = message.match(SIZE_PATTERN);
  if (!match) return null;
  const value = parseFloat(match[1]);
  const unit = match[2].toLowerCase();
  if (unit.startsWith("mm") || unit.startsWith("millimet")) return value / 10;
  if (unit.startsWith("cm") || unit.startsWith("centimet")) return value;
  if (unit === "m" || unit.startsWith("meter")) return value * 100;
  if (unit.startsWith("in")) return value * 2.54;
  if (unit.startsWith("ft") || unit.startsWith("feet")) return value * 30.48;
  return null;
}

// Vague scale phrasing has no exact factor, so these are reasoned
// defaults, not measurements — "a bit" implies a smaller nudge than a
// bare "bigger"/"smaller". An explicit number ("2x", "by 50%") always
// wins over a vague word when both could apply.
function extractScaleFactor(message: string): number | null {
  const explicitMultiplier = message.match(/(\d+(?:\.\d+)?)\s*(x|times)\b/i);
  if (explicitMultiplier) return parseFloat(explicitMultiplier[1]);

  const byPercent = message.match(/by\s+(\d+(?:\.\d+)?)\s*%/i);
  if (byPercent) {
    const delta = parseFloat(byPercent[1]) / 100;
    return /smaller/i.test(message) ? 1 - delta : 1 + delta;
  }

  if (/twice|double/i.test(message)) return 2.0;
  if (/half/i.test(message)) return 0.5;
  if (/\bbit\b.*bigger|bigger.*\bbit\b|\bbit\b.*larger|larger.*\bbit\b/i.test(message)) return 1.2;
  if (/\bbit\b.*smaller|smaller.*\bbit\b/i.test(message)) return 0.8;
  if (/bigger|larger/i.test(message)) return 1.5;
  if (/smaller/i.test(message)) return 0.67;
  return null;
}

function extractMirrorAxis(message: string): "x" | "y" | "z" {
  if (/vertical|top[\s-]?to[\s-]?bottom|up[\s-]?down/i.test(message)) return "y";
  if (/depth|front[\s-]?to[\s-]?back/i.test(message)) return "z";
  // Left-right is both the default and the most common request — mirror
  // patterns like "flip it"/"mirror it" alone give no axis hint at all.
  return "x";
}

// Confirmed live: real phrasing for face enrollment varies a lot more than
// a single "remember this face AS X" construction covers — "name is X",
// "this is X", "it's X", "called X" are all natural ways to say the same
// thing, and a request phrased that way was silently swallowed by plain
// chat before (the model just improvised a friendly-sounding
// acknowledgment without ever actually calling /enroll). Tries each
// construction in turn and returns the first name found; capped at two
// words (first + last name) to limit how much of a run-on sentence a
// missing comma could pull in.
function extractEnrollName(message: string): string | null {
  const NAME = "([A-Za-z][A-Za-z'-]*(?:\\s+[A-Za-z][A-Za-z'-]*){0,1})";
  const patterns = [
    new RegExp(`\\bas\\s+${NAME}`, "i"),
    new RegExp(`\\bname(?:'s| is)\\s+${NAME}`, "i"),
    new RegExp(`\\b(?:this is|it'?s|it is|he'?s|he is|she'?s|she is)\\s+${NAME}`, "i"),
    new RegExp(`\\bcalled\\s+${NAME}`, "i")
  ];
  for (const pattern of patterns) {
    const match = message.match(pattern);
    if (match) return match[1].trim();
  }
  return null;
}

function extractSimplifyPercent(message: string): number {
  const explicit = message.match(/by\s+(\d+(?:\.\d+)?)\s*%/i);
  if (explicit) return 100 - parseFloat(explicit[1]);
  if (/half/i.test(message)) return 50;
  // No percent mentioned at all — 50% is a reasonable, visible-but-not-
  // destructive default reduction.
  return 50;
}

// Rescales the generated mesh (via trimesh, already installed for
// Hunyuan3D-2) so its largest bounding-box dimension matches the requested
// size, before it ever reaches Fusion — sidesteps relying on Fusion's own
// scale tooling supporting mesh bodies, which isn't clearly documented
// either way. Writes to a separate output file rather than overwriting the
// input: if this step fails partway, the original mesh must still be
// importable as a fallback.
function scaleMeshToTargetSize(meshPath: string, targetSizeCm: number): Promise<string | null> {
  return new Promise((resolve) => {
    const scaledPath = meshPath.replace(/\.obj$/i, "_scaled.obj");
    const child = spawn(HUNYUAN3D_PYTHON, [path.join(IMAGE23D_DIR, "scale_mesh.py"), meshPath, String(targetSizeCm), scaledPath]);
    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", (err) => {
      console.error("Failed to launch mesh scaling step:", err.message);
      resolve(null);
    });
    child.on("exit", (code) => {
      if (code !== 0 || !fs.existsSync(scaledPath)) {
        console.error("Mesh scaling step failed:", stderr.trim().slice(-1000));
        resolve(null);
        return;
      }
      resolve(scaledPath);
    });
  });
}

// Converts the final (post-scaling, if any) mesh to .glb purely for the
// browser "view model" panel — Fusion needs the .obj, <model-viewer> needs
// glTF, so this runs as an extra step alongside the Fusion import rather
// than replacing it. Best-effort: a failure here shouldn't sink the whole
// reply just because the preview couldn't be generated.
function convertMeshToGlb(meshPath: string, outputPath: string): Promise<boolean> {
  // MODELS_DIR otherwise only gets created inside saveModelsIndex(), which
  // doesn't run until AFTER this succeeds — confirmed directly: the very
  // first conversion failed because its own output directory didn't exist
  // yet. Same class of bug as saveHistory()/writeFile() earlier.
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  return new Promise((resolve) => {
    const child = spawn(HUNYUAN3D_PYTHON, [path.join(IMAGE23D_DIR, "convert_to_glb.py"), meshPath, outputPath]);
    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", (err) => {
      console.error("Failed to launch GLB conversion step:", err.message);
      resolve(false);
    });
    child.on("exit", (code) => {
      if (code !== 0 || !fs.existsSync(outputPath)) {
        console.error("GLB conversion step failed:", stderr.trim().slice(-1000));
        resolve(false);
        return;
      }
      resolve(true);
    });
  });
}

// Runs a single-mesh edit operation (scale-factor/mirror/simplify) via
// edit_mesh.py. Writes to a separate output file — never overwrites the
// input — same reasoning as scaleMeshToTargetSize/repairMesh: a failed
// edit must leave the original mesh intact.
function editMesh(
  meshPath: string,
  outputPath: string,
  operation: "scale-factor" | "mirror" | "simplify" | "simplify-percent",
  param?: string
): Promise<boolean> {
  return new Promise((resolve) => {
    const args = [path.join(IMAGE23D_DIR, "edit_mesh.py"), meshPath, outputPath, operation];
    if (param !== undefined) args.push(param);
    const child = spawn(HUNYUAN3D_PYTHON, args);
    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", (err) => {
      console.error("Failed to launch mesh edit step:", err.message);
      resolve(false);
    });
    child.on("exit", (code) => {
      if (code !== 0 || !fs.existsSync(outputPath)) {
        console.error("Mesh edit step failed:", stderr.trim().slice(-1000));
        resolve(false);
        return;
      }
      resolve(true);
    });
  });
}

interface MeshRepairReport {
  watertightBefore: boolean;
  watertightAfter: boolean;
  componentsFound: number;
  componentDroppedCount: number;
  repaired: boolean;
  usable: boolean;
  reason: string | null;
}

// Validates the raw generation output before it's trusted for Fusion import
// or slicing — Hunyuan3D-2 occasionally produces a "detached limb" mesh
// (an unrelated stray blob disconnected from the main shape), which passes
// a naive watertight check per-component but is still garbage. Returns
// null if the repair step itself couldn't run (script crash) — that's
// "couldn't validate", not "confirmed broken", so callers should treat it
// as a soft failure and fall back to the unrepaired mesh rather than
// blocking the whole pipeline on a tooling bug. report.usable === false is
// the actual "this mesh is too broken" verdict.
function repairMesh(meshPath: string): Promise<{ outputPath: string; report: MeshRepairReport } | null> {
  return new Promise((resolve) => {
    const repairedPath = meshPath.replace(/\.obj$/i, "_repaired.obj");
    const child = spawn(HUNYUAN3D_PYTHON, [path.join(IMAGE23D_DIR, "repair_mesh.py"), meshPath, repairedPath]);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => { stdout += d.toString(); });
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", (err) => {
      console.error("Failed to launch mesh repair step:", err.message);
      resolve(null);
    });
    child.on("exit", (code) => {
      if (code !== 0) {
        console.error("Mesh repair step failed:", stderr.trim().slice(-1000));
        resolve(null);
        return;
      }
      let parsed: any;
      try {
        parsed = JSON.parse(stdout.trim().split("\n").pop() || "");
      } catch {
        console.error("Mesh repair step produced unparseable output:", stdout.trim().slice(-1000));
        resolve(null);
        return;
      }
      const report: MeshRepairReport = {
        watertightBefore: parsed.watertight_before,
        watertightAfter: parsed.watertight_after,
        componentsFound: parsed.components_found,
        componentDroppedCount: parsed.component_dropped_count,
        repaired: parsed.repaired,
        usable: parsed.usable,
        reason: parsed.reason ?? null,
      };
      if (report.usable && !fs.existsSync(repairedPath)) {
        console.error("Mesh repair step reported usable but output file is missing:", repairedPath);
        resolve(null);
        return;
      }
      resolve({ outputPath: report.usable ? repairedPath : "", report });
    });
  });
}

interface SliceResult {
  success: boolean;
  gcodePath?: string;
  estimatedTimeMin?: number;
  estimatedFilamentG?: number;
  error?: string;
  knownCliBug?: boolean;
}

// Extracts the header of Metadata/plate_1.gcode from inside the sliced
// .gcode.3mf (itself a zip archive) via Windows' built-in tar, which can
// read zip entries directly — avoids adding a zip-parsing dependency for
// two lines of text. Reads by streaming and stopping early rather than
// buffering the whole file: a real print's gcode body can be many MB, but
// the header lines this needs appear in the first ~20 lines. Confirmed
// live against a real Bambu Studio CLI output: the header contains exactly
// `; total estimated time: <Xh Ym Zs>` and
// `; total filament weight [g] : <N>`.
// Uses an absolute path to Windows' native tar.exe rather than relying on
// PATH — confirmed live that Git for Windows' bundled tar (usually earlier
// on PATH than System32) fails on Windows-style paths with a "cannot
// connect to C: resolve failed" error, since it parses "C:" as a remote
// host spec the way Unix tar does. System32's real tar.exe doesn't have
// that problem.
const WINDOWS_TAR_EXE = "C:/Windows/System32/tar.exe";

function parseSliceEstimate(gcode3mfPath: string): Promise<{ estimatedTimeMin: number | null; estimatedFilamentG: number | null }> {
  return new Promise((resolve) => {
    const child = spawn(WINDOWS_TAR_EXE, ["-xf", gcode3mfPath, "-O", "Metadata/plate_1.gcode"]);
    let buffer = "";
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      child.kill();

      const timeMatch = buffer.match(/total estimated time:\s*([\dhms\s]+?)(?:\r?\n|$)/i);
      let estimatedTimeMin: number | null = null;
      if (timeMatch) {
        const h = timeMatch[1].match(/(\d+)h/);
        const m = timeMatch[1].match(/(\d+)m/);
        const s = timeMatch[1].match(/(\d+)s/);
        if (h || m || s) {
          const totalSeconds = (h ? parseInt(h[1]) * 3600 : 0) + (m ? parseInt(m[1]) * 60 : 0) + (s ? parseInt(s[1]) : 0);
          estimatedTimeMin = Math.round(totalSeconds / 60);
        }
      }
      const weightMatch = buffer.match(/total filament weight \[g\]\s*:\s*([\d.]+)/i);
      const estimatedFilamentG = weightMatch ? parseFloat(weightMatch[1]) : null;

      resolve({ estimatedTimeMin, estimatedFilamentG });
    };
    // tar's own exit code isn't reliable here (confirmed live: it can exit
    // non-zero via -O single-entry extraction while still having written
    // valid data to stdout) — success is judged by whether the expected
    // fields were actually found in what was read, not the exit code.
    child.stdout.on("data", (d) => {
      buffer += d.toString();
      if (buffer.length > 4000) finish();
    });
    child.on("error", () => resolve({ estimatedTimeMin: null, estimatedFilamentG: null }));
    child.on("close", finish);
    setTimeout(finish, 5000);
  });
}

// Slices a mesh via Bambu Studio's own CLI (not OrcaSlicer, not GUI
// automation — see PRINTER_SETUP.md for why). Confirmed live: the CLI
// accepts a raw .obj directly (no .3mf wrapper needed), and this exact
// profile/argument combination does not trigger the known P2S
// "nozzle_volume_type" CLI bug — that check below still guards against it
// for other profile combinations that might.
function sliceModel(meshPath: string): Promise<SliceResult> {
  return new Promise((resolve) => {
    if (!fs.existsSync(BAMBU_STUDIO_EXE)) {
      resolve({ success: false, error: `Bambu Studio not found at ${BAMBU_STUDIO_EXE}. Set BAMBU_STUDIO_EXE in .env, or see PRINTER_SETUP.md.` });
      return;
    }
    if (!BAMBU_MACHINE_PROFILE || !BAMBU_PROCESS_PROFILE || !BAMBU_FILAMENT_PROFILE) {
      resolve({ success: false, error: "Bambu Studio profile paths aren't configured — set BAMBU_MACHINE_PROFILE/BAMBU_PROCESS_PROFILE/BAMBU_FILAMENT_PROFILE in .env (see PRINTER_SETUP.md)." });
      return;
    }

    const gcodePath = meshPath.replace(/\.(obj|3mf)$/i, ".gcode.3mf");
    const args = [
      "--slice", "0",
      "--load-settings", `${BAMBU_MACHINE_PROFILE};${BAMBU_PROCESS_PROFILE}`,
      "--load-filaments", BAMBU_FILAMENT_PROFILE,
      "--export-3mf", gcodePath,
      meshPath,
    ];

    // Confirmed live: the CLI writes a result.json (slice stats) into its
    // own working directory as a side effect of --export-3mf — without an
    // explicit cwd it landed in the repo root. Pointing it at the same
    // directory as the output file keeps it alongside the other
    // already-gitignored generation artifacts instead.
    const child = spawn(BAMBU_STUDIO_EXE, args, { cwd: path.dirname(gcodePath) });
    let stderr = "";
    let stdout = "";
    child.stdout.on("data", (d) => { stdout += d.toString(); });
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", (err) => {
      resolve({ success: false, error: `Failed to launch Bambu Studio CLI: ${err.message}` });
    });
    child.on("exit", async (code) => {
      if (code !== 0 || !fs.existsSync(gcodePath)) {
        const combined = (stderr + stdout).slice(-2000);
        // A reported open GitHub issue on P2S CLI slicing specifically —
        // distinguish it from a generic failure so the reply is actionable
        // instead of an opaque "slicing failed."
        const knownCliBug = /nozzle_volume_type/i.test(combined);
        resolve({
          success: false,
          error: knownCliBug
            ? 'Bambu Studio\'s CLI slicer hit a known bug on this printer profile ("nozzle_volume_type" error) — see PRINTER_SETUP.md\'s Known Limitations. Slice this one manually in Bambu Studio instead.'
            : `Slicing failed: ${combined || "no output captured"}`,
          knownCliBug,
        });
        return;
      }
      const estimate = await parseSliceEstimate(gcodePath);
      resolve({
        success: true,
        gcodePath,
        ...(estimate.estimatedTimeMin !== null ? { estimatedTimeMin: estimate.estimatedTimeMin } : {}),
        ...(estimate.estimatedFilamentG !== null ? { estimatedFilamentG: estimate.estimatedFilamentG } : {}),
      });
    });
  });
}

interface GeneratedModelEntry {
  id: string;
  subject: string;
  filename: string;
  createdAt: string;
  saved: boolean;
}

function loadModelsIndex(): GeneratedModelEntry[] {
  try {
    return JSON.parse(fs.readFileSync(MODELS_INDEX_FILE, "utf8"));
  } catch {
    return [];
  }
}

function saveModelsIndex(entries: GeneratedModelEntry[]): void {
  fs.mkdirSync(MODELS_DIR, { recursive: true });
  fs.writeFileSync(MODELS_INDEX_FILE, JSON.stringify(entries, null, 2));
}

// Turns a subject like "rubber duck" into "rubber-duck" for use in a
// filename — used when a model gets saved (see the /save endpoint), not at
// generation time, since the raw timestamp filename is fine until a human
// might actually go looking for the file directly.
function slugify(text: string): string {
  const slug = text.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || "model";
}

// Registers a freshly-converted .glb, then prunes the oldest UNSAVED
// entries beyond MAX_UNSAVED_MODELS so preview files don't accumulate
// forever with regular use — saved models are exempt from this entirely.
function registerGeneratedModel(subject: string, filename: string): GeneratedModelEntry {
  const entries = loadModelsIndex();
  const entry: GeneratedModelEntry = {
    id: randomUUID(),
    subject,
    filename,
    createdAt: new Date().toISOString(),
    saved: false
  };
  entries.push(entry);

  const unsaved = entries.filter(e => !e.saved);
  if (unsaved.length > MAX_UNSAVED_MODELS) {
    const toPrune = unsaved
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt))
      .slice(0, unsaved.length - MAX_UNSAVED_MODELS);
    const pruneIds = new Set(toPrune.map(e => e.id));
    for (const old of toPrune) {
      fs.rm(path.join(MODELS_DIR, old.filename), { force: true }, () => {});
    }
    saveModelsIndex(entries.filter(e => !pruneIds.has(e.id)));
  } else {
    saveModelsIndex(entries);
  }

  return entry;
}

// Resolves "the model"/"it" in a mesh-edit request to a real registry
// entry. Prefers whichever model the frontend says is currently open in
// the preview panel (openModelId) — the most reliable signal, since it's
// exactly what the user is looking at — and falls back to the most
// recently created entry when nothing is open (e.g. the user says "make
// it bigger" right after a generation, before ever opening the panel).
function resolveEditTargetModel(openModelId?: string | null): GeneratedModelEntry | null {
  const entries = loadModelsIndex();
  if (openModelId) {
    const explicit = entries.find(e => e.id === openModelId);
    if (explicit) return explicit;
  }
  if (entries.length === 0) return null;
  return entries.reduce((newest, e) => (e.createdAt > newest.createdAt ? e : newest));
}

// Swaps a registry entry's file after a successful edit — same id, new
// filename — so SAVE/DISCARD and the viewer panel's "always refetch by
// id" pattern (openModelViewerPanelById) keep working with zero frontend
// changes. Deletes the pre-edit file only after the new one is confirmed
// on disk. saved/id/subject are left untouched, so an already-saved model
// stays saved through an edit.
function replaceGeneratedModelFile(id: string, newAbsolutePath: string): GeneratedModelEntry | null {
  const entries = loadModelsIndex();
  const entry = entries.find(e => e.id === id);
  if (!entry) return null;
  const oldFilename = entry.filename;
  const newFilename = `${Date.now()}.glb`;
  fs.renameSync(newAbsolutePath, path.join(MODELS_DIR, newFilename));
  entry.filename = newFilename;
  saveModelsIndex(entries);
  fs.rm(path.join(MODELS_DIR, oldFilename), { force: true }, () => {});
  return entry;
}

// SearXNG's image category, used to find a reference photo for a named
// subject before running it through the local image-to-3D model.
async function webImageSearch(query: string): Promise<string[]> {
  try {
    const url = `${SEARXNG_URL}/search?q=${encodeURIComponent(query)}&format=json&categories=images`;
    const res = await fetch(url);

    if (!res.ok) {
      console.error("SearXNG image search failed:", res.status, await res.text().catch(() => ""));
      return [];
    }

    const data = await res.json() as { results?: any[] };
    return (data.results ?? [])
      .map((r: any) => r.img_src as string)
      // SearXNG's image category mixes in unrelated dev-icon SVGs as noise
      // (confirmed directly: devicons/lucide-static entries showing up for
      // completely unrelated queries) — Hunyuan3D and the vision-model
      // verification step both need a real raster photo, and Ollama flatly
      // rejects SVGs ("Failed to load image or audio file"), so there's no
      // point letting them occupy a candidate slot.
      .filter((src: unknown): src is string => typeof src === "string" && /^https?:\/\//.test(src) && !/\.svg(\?|$)/i.test(src));
  } catch (err) {
    console.error("Web image search error (is SearXNG running at " + SEARXNG_URL + "?):", err);
    return [];
  }
}

// Best-effort cleanup for the reference image and intermediate mesh files
// created per image-to-3D request — this path is the only thing writing to
// IMAGE23D_TMP_DIR, and without this it grows unbounded since nothing else
// ever revisits old requests. Swallows errors since a leftover file here is
// harmless and the request has already been answered by the time this runs.
function cleanupTempFiles(paths: (string | null | undefined)[]): void {
  for (const p of paths) {
    if (!p) continue;
    fs.rm(p, { force: true }, (err) => {
      if (err) console.error("Temp file cleanup failed:", p, err.message);
    });
  }
}

async function downloadImageToFile(imageUrl: string, destPath: string): Promise<boolean> {
  try {
    // Some image hosts reject requests with no browser-like User-Agent
    // (hotlink protection) — SearXNG's image results come from arbitrary
    // sites, so this needs to look like a normal browser fetch.
    const res = await fetch(imageUrl, {
      signal: AbortSignal.timeout(15000),
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" }
    });
    if (!res.ok) {
      console.error("Reference image download failed:", imageUrl, "HTTP", res.status);
      return false;
    }
    // Belt-and-suspenders alongside the .svg filter in webImageSearch — a
    // URL with no telltale extension can still serve SVG, HTML (a soft
    // 404), or another format neither Hunyuan3D nor Ollama's vision model
    // can decode, and that's cheaper to catch here than after a wasted
    // verification round-trip.
    const contentType = res.headers.get("content-type") ?? "";
    if (!contentType.startsWith("image/") || contentType.includes("svg")) {
      console.error("Reference image download rejected:", imageUrl, "content-type", contentType);
      return false;
    }
    const buffer = Buffer.from(await res.arrayBuffer());
    fs.mkdirSync(path.dirname(destPath), { recursive: true });
    fs.writeFileSync(destPath, buffer);
    return true;
  } catch (err) {
    console.error("Reference image download error:", imageUrl, err);
    return false;
  }
}

// Tries each candidate URL in turn until one actually downloads — a single
// dead link (404, hotlink protection, or a domain your DNS resolver blocks —
// confirmed directly: a Pi-hole install killed teahub.io and nothing else
// downstream noticed) shouldn't sink the whole search when SearXNG handed
// back several other usable results (typically hundreds). Capped well below
// that since there's no value in exhausting all of them — 10 rather than 5
// specifically because the multiview path's verify callback checks both
// subject match AND full-body framing (confirmed directly: that two-part
// bar has a real rejection rate, so 5 tries wasn't always enough headroom).
async function downloadFirstAvailableImage(urls: string[], destPath: string, verify?: (path: string) => Promise<boolean>): Promise<string | null> {
  for (const url of urls.slice(0, 10)) {
    if (await downloadImageToFile(url, destPath)) {
      if (!verify || await verify(destPath)) return url;
    }
  }
  return null;
}

// Saves a UI-attached image (sent as base64 to keep the client-side upload
// simple, no multipart parsing needed) to a temp file for the image-to-3D
// model to read.
function saveBase64ImageToFile(base64: string, mimeType: string): string {
  const ext = (mimeType.split("/")[1] ?? "jpg").replace("jpeg", "jpg");
  const filePath = path.join(IMAGE23D_TMP_DIR, `${Date.now()}.${ext}`);
  fs.mkdirSync(IMAGE23D_TMP_DIR, { recursive: true });
  fs.writeFileSync(filePath, Buffer.from(base64, "base64"));
  return filePath;
}

interface MeshGenerationResult {
  success: boolean;
  meshPath?: string;
  error?: string;
}

// Real generative model with texture-capable output (though this call only
// uses its shape stage; see HUNYUAN3D_DIR's comment above). The first call
// downloads several GB of weights from Hugging Face, hence the long timeout.
function generateMeshWithHunyuan3D(imagePath: string): Promise<MeshGenerationResult> {
  return new Promise((resolve) => {
    if (!fs.existsSync(HUNYUAN3D_PYTHON)) {
      resolve({ success: false, error: `Hunyuan3D venv not found at ${HUNYUAN3D_PYTHON}.` });
      return;
    }

    // Must be .obj (or .stl/.3mf) — Fusion's meshBodies.add() only accepts
    // those three formats and rejects .glb with "Unsupported file format"
    // (confirmed directly). trimesh infers export format from the
    // extension, so this alone is enough to fix it.
    const outputPath = path.join(IMAGE23D_TMP_DIR, `${Date.now()}_hunyuan.obj`);
    const child = spawn(HUNYUAN3D_PYTHON, ["run_shape_only.py", imagePath, outputPath], {
      cwd: HUNYUAN3D_DIR,
      // Hugging Face's newer "Xet" transfer backend has a well-documented
      // stalling bug on large files (confirmed directly — a multi-GB weight
      // file hung indefinitely at 0 bytes). Disabling it falls back to the
      // older, reliable HTTP downloader. Only matters for the first run
      // per model (weights are cached locally after), but costs nothing to
      // always set.
      env: { ...process.env, HF_HUB_DISABLE_XET: "1" }
    });

    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });

    const timeout = setTimeout(() => {
      child.kill();
      resolve({ success: false, error: "Hunyuan3D generation timed out after 8 minutes." });
    }, 480000);

    child.on("error", (err) => {
      clearTimeout(timeout);
      resolve({ success: false, error: `Failed to launch Hunyuan3D: ${err.message}` });
    });

    child.on("exit", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        resolve({ success: false, error: stderr.trim().slice(-2000) || `Hunyuan3D exited with code ${code}` });
        return;
      }
      if (!fs.existsSync(outputPath)) {
        resolve({ success: false, error: "Hunyuan3D finished but no mesh file was produced." });
        return;
      }
      resolve({ success: true, meshPath: outputPath });
    });
  });
}

// Experimental: feeds three independently web-searched angle photos (front/
// left/back) into Hunyuan3D-2mv instead of one photo into shape-only. Real
// multiview conditioning gives better geometry ONLY if the three images
// actually agree on what they're depicting — since they come from separate
// searches, not one coordinated photoshoot, that's not guaranteed (different
// art style, outfit variant, or pose per image is a real risk). Downloads
// its own multi-GB weights from tencent/Hunyuan3D-2mv on first run, on top
// of the shape-only model's weights.
function generateMeshWithHunyuan3DMultiview(frontPath: string, leftPath: string, backPath: string): Promise<MeshGenerationResult> {
  return new Promise((resolve) => {
    if (!fs.existsSync(HUNYUAN3D_PYTHON)) {
      resolve({ success: false, error: `Hunyuan3D venv not found at ${HUNYUAN3D_PYTHON}.` });
      return;
    }

    const outputPath = path.join(IMAGE23D_TMP_DIR, `${Date.now()}_hunyuan_mv.obj`);
    const child = spawn(HUNYUAN3D_PYTHON, ["run_multiview.py", frontPath, leftPath, backPath, outputPath], {
      cwd: HUNYUAN3D_DIR,
      env: { ...process.env, HF_HUB_DISABLE_XET: "1" }
    });

    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });

    const timeout = setTimeout(() => {
      child.kill();
      // 45 minutes, not 10 — the first-ever call downloads its own multi-GB
      // weight set from tencent/Hunyuan3D-2mv (confirmed directly: the base
      // Hunyuan3D-2 weights alone took ~30 minutes on this connection), on
      // top of whatever the actual diffusion pass needs afterward.
      resolve({ success: false, error: "Hunyuan3D multiview generation timed out after 45 minutes." });
    }, 2700000);

    child.on("error", (err) => {
      clearTimeout(timeout);
      resolve({ success: false, error: `Failed to launch Hunyuan3D multiview: ${err.message}` });
    });

    child.on("exit", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        resolve({ success: false, error: stderr.trim().slice(-2000) || `Hunyuan3D multiview exited with code ${code}` });
        return;
      }
      if (!fs.existsSync(outputPath)) {
        resolve({ success: false, error: "Hunyuan3D multiview finished but no mesh file was produced." });
        return;
      }
      resolve({ success: true, meshPath: outputPath });
    });
  });
}

// Mirrors executeFusionScript's shape, but hits the bridge's mesh-import
// endpoint instead of exec()'ing generated code.
async function importMeshToFusion(meshPath: string): Promise<FusionExecutionResult> {
  try {
    const res = await fetch(`${FUSION_BRIDGE_URL}/import-mesh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: meshPath }),
      signal: AbortSignal.timeout(35000)
    });

    if (!res.ok) {
      return { success: false, error: `Fusion bridge returned HTTP ${res.status}` };
    }

    return await res.json() as FusionExecutionResult;
  } catch (err: any) {
    return {
      success: false,
      error: `Could not reach the Fusion 360 bridge at ${FUSION_BRIDGE_URL}: ${err.message}. ` +
        `Is Fusion 360 running with the NoahFusionBridge add-in active?`
    };
  }
}

// Real system telemetry for the HUD, replacing what used to be Math.random()
// fake fluctuations — COGNITION/DATA GRID/SPEECH reflect whether the backing
// services (Ollama, SearXNG, the speech service respectively) are actually
// reachable, NEURAL LOAD / MODEL TEMP read genuine GPU utilization and
// temperature, and MESH_GEN combines Fusion bridge reachability with
// gpuExclusiveTaskRunning (offline/idle/generating — see the frontend for
// how those combine). Uses isServiceHealthy directly rather than
// ensureFusionBridgeAvailable() — that function auto-launches Fusion if
// it's not running, which would make this 3-second-polled endpoint launch
// Fusion 360 (and block for up to 60s) just from having the HUD open.
app.get('/api/hud-metrics', async (_, res) => {
  const [ollamaHealthy, searxngHealthy, speechHealthy, fusionHealthy] = await Promise.all([
    isServiceHealthy(OLLAMA_URL.replace(/\/api\/generate$/, "/api/version")),
    isServiceHealthy(`${SEARXNG_URL}/healthz`),
    isServiceHealthy(`${SPEECH_SERVICE_URL}/health`),
    isServiceHealthy(`${FUSION_BRIDGE_URL}/health`)
  ]);

  const services = { ollama: ollamaHealthy, searxng: searxngHealthy, speech: speechHealthy };
  const healthyCount = Object.values(services).filter(Boolean).length;
  const coreIntegrityPercent = (healthyCount / Object.keys(services).length) * 100;

  // Soft-fail, same as every health check above — if the vision service is
  // down or disabled (its default), this endpoint must not break the HUD.
  // The greeting text is built here, not on the frontend, so it's identical
  // whether it ends up spoken, displayed, or read back from history later.
  let proactiveGreeting: string | null = null;
  try {
    const eventRes = await fetch(`${VISION_SERVICE_URL}/pending-event`, { signal: AbortSignal.timeout(1500) });
    if (eventRes.ok) {
      const event = await eventRes.json() as { type: "known" | "unknown"; name: string | null } | null;
      if (event) {
        proactiveGreeting = event.type === "known"
          ? `Welcome back, ${event.name}!`
          : "Someone I don't recognize just showed up.";
        conversationHistory.push(`Assistant: ${proactiveGreeting}`);
        saveHistory();
      }
    }
  } catch {
    // Vision service unreachable — nothing to announce this tick.
  }

  res.json({
    coreIntegrityPercent,
    generating: gpuExclusiveTaskRunning,
    fusionAvailable: fusionHealthy,
    services,
    printer: { installed: isBambuConnectInstalled() },
    gpu: getGpuStats(),
    proactiveGreeting
  });
});

// API: real, live GPU stats for the system panel's hardware section —
// same getGpuStats() the chat-based "what's my GPU" intent uses, just as a
// plain REST endpoint the frontend can poll on panel-open without needing
// to phrase a chat message.
app.get('/api/hardware', (_, res) => {
  res.json({ gpu: getGpuStats() });
});

// API: the system panel's model-recommendation section — real pulled-model
// data (buildModelRecommendationData, shared with the chat intent) plus a
// short LLM-generated pros/cons blurb for each candidate versus CHAT_MODEL
// (the model actually used for a plain chat turn). The pros/cons layer is
// enrichment, not core data: if the LLM call fails or returns unparseable
// JSON, `analysis` comes back null and the frontend still renders the real
// model list with fit badges — matching this file's established "never
// fail hard over an optional layer" convention (e.g. listPulledModels()
// returning [] rather than throwing).
// MUST be registered before /api/models/:id below — Express matches routes
// in registration order, and :id matches any literal segment (including
// "overview"), so the wildcard route would otherwise swallow this one.
app.get('/api/models/overview', async (_, res) => {
  const gpu = getGpuStats();
  const pulled = await listPulledModels();
  const recommendation = buildModelRecommendationData(pulled, gpu);

  // Deliberately excludes `suggestions` — confirmed live that including
  // both pulled models and the full suggestion table (17 candidates total
  // on a typical setup) produced consistently truncated/malformed JSON
  // within a reasonable num_predict budget. Pros/cons for models you'd
  // actually compare against (already pulled) is the higher-value case
  // anyway; suggestions still show in the list, just without this layer.
  const candidates = recommendation.pulled
    .filter(m => m.name !== CHAT_MODEL)
    .map(m => ({ name: m.name, detail: `${m.sizeGB.toFixed(1)}GB, already pulled, ${m.fits === false ? "exceeds your VRAM" : "fits your VRAM"}` }));

  let analysis: { model: string; pros: string[]; cons: string[] }[] | null = null;
  if (candidates.length > 0) {
    const analysisPrompt = `Noah's default day-to-day chat model is "${CHAT_MODEL}". For each of the following candidate models, give 1-3 short pros and 1-3 short cons versus ${CHAT_MODEL} specifically (quality, speed, size/VRAM cost, use-case fit). Candidates:\n${candidates.map(c => `- ${c.name} (${c.detail})`).join("\n")}\n\nReply with ONLY a JSON array, no other text, in exactly this shape: [{"model": "name", "pros": ["..."], "cons": ["..."]}, ...] — one entry per candidate listed above, using its exact name.`;
    // Non-deterministic output (temperature 0.3) occasionally comes back as
    // near-valid-but-malformed JSON (confirmed live: a missing closing
    // bracket on the last "cons" array) — one retry is cheap and usually
    // produces a parseable result on the second attempt; if both fail,
    // `analysis` stays null and the model cards render without pros/cons,
    // per the established "enrichment, not core data" fallback.
    for (let attempt = 0; attempt < 2 && analysis === null; attempt++) {
      try {
        const raw = await callOllamaModel(analysisPrompt, 600, CHAT_MODEL, 45000);
        const jsonMatch = raw.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          if (Array.isArray(parsed)) analysis = parsed;
        }
      } catch (err: any) {
        console.warn(`Model overview pros/cons generation failed on attempt ${attempt + 1} (non-fatal):`, err.message);
      }
    }
  }

  res.json({ currentModel: CHAT_MODEL, hardware: gpu, ...recommendation, analysis });
});

// API: returns a model's current info by id — used by the frontend instead
// of trusting a URL baked into a chat message's button at creation time,
// since /save below can rename the file out from under that URL later.
app.get('/api/models/:id', (req, res) => {
  const entry = loadModelsIndex().find(e => e.id === req.params.id);
  if (!entry) return res.status(404).json({ success: false, error: "Model not found — it may already have been pruned or discarded." });
  res.json({ success: true, id: entry.id, subject: entry.subject, url: `/generated-models/${entry.filename}`, saved: entry.saved });
});

// API: exempts a generated model from the MAX_UNSAVED_MODELS rotation —
// the "SAVE" button in the model viewer panel. Also renames the file from
// its generation-time timestamp to something derived from the subject
// (e.g. rubber-duck.glb), purely so the file means something if you ever
// go looking for it directly outside the app — nothing in the UI itself
// shows the raw filename.
app.post('/api/models/:id/save', (req, res) => {
  const entries = loadModelsIndex();
  const entry = entries.find(e => e.id === req.params.id);
  if (!entry) return res.status(404).json({ success: false, error: "Model not found — it may already have been pruned or discarded." });

  const baseSlug = slugify(entry.subject);
  let newFilename = `${baseSlug}.glb`;
  let suffix = 2;
  while (
    entries.some(e => e.id !== entry.id && e.filename === newFilename) ||
    (newFilename !== entry.filename && fs.existsSync(path.join(MODELS_DIR, newFilename)))
  ) {
    newFilename = `${baseSlug}-${suffix}.glb`;
    suffix++;
  }

  if (newFilename !== entry.filename) {
    try {
      fs.renameSync(path.join(MODELS_DIR, entry.filename), path.join(MODELS_DIR, newFilename));
      entry.filename = newFilename;
    } catch (err: any) {
      console.error("Failed to rename saved model file:", err.message);
      // Not fatal — still mark it saved under its existing filename rather
      // than losing the save entirely over a cosmetic rename failing.
    }
  }

  entry.saved = true;
  saveModelsIndex(entries);
  res.json({ success: true, url: `/generated-models/${entry.filename}` });
});

// API: immediately deletes a generated model — the "DISCARD" button.
app.delete('/api/models/:id', (req, res) => {
  const entries = loadModelsIndex();
  const entry = entries.find(e => e.id === req.params.id);
  if (!entry) return res.status(404).json({ success: false, error: "Model not found — it may already have been pruned or discarded." });
  fs.rm(path.join(MODELS_DIR, entry.filename), { force: true }, (err) => {
    if (err) console.error("Failed to delete discarded model file:", entry.filename, err.message);
  });
  saveModelsIndex(entries.filter(e => e.id !== req.params.id));
  res.json({ success: true });
});

// API: the system panel's live comparison section — same runModelComparison
// the chat-based "compare X and Y on: ..." intent uses, but with the model
// names and prompt supplied directly by the UI instead of extracted from a
// natural-language message. No auto-pull, matching the chat feature's
// behavior — a model not already on disk is a clean rejection, not a
// multi-GB/multi-minute pull blocking the request.
app.post('/api/models/compare', async (req, res) => {
  const { modelA, modelB, prompt } = req.body as { modelA?: string; modelB?: string; prompt?: string };
  if (!modelA || !modelB || !prompt) {
    return res.status(400).json({ error: "modelA, modelB, and prompt are all required." });
  }

  const pulled = await listPulledModels();
  const pulledNames = new Set(pulled.map(m => m.name));
  const missing = [modelA, modelB].filter(m => !pulledNames.has(m));
  if (missing.length > 0) {
    return res.status(400).json({ error: `Not pulled: ${missing.join(", ")}`, available: pulled.map(m => m.name) });
  }

  const comparison = await runModelComparison(modelA, modelB, prompt);
  res.json({ comparison });
});

// API: extracts, chunks, and embeds an attached document, storing it in
// documentStore for cross-turn Q&A (see looksLikeDocumentQuestion in
// /api/chat). Base64-in-JSON, matching the existing image-attach
// convention, rather than adding multipart/multer handling for this one
// endpoint.
app.post('/api/documents/upload', async (req, res) => {
  const { base64, mimeType, fileName } = req.body ?? {};
  if (!base64 || !fileName) {
    return res.status(400).json({ error: "Missing file data." });
  }

  try {
    const text = await extractDocumentText(base64, mimeType, fileName);
    if (!text.trim()) {
      return res.status(400).json({ error: "No extractable text found in that file." });
    }
    if (text.length > MAX_DOCUMENT_CHARS) {
      return res.status(400).json({ error: `Document is too large (${text.length} characters, limit ${MAX_DOCUMENT_CHARS}).` });
    }

    const chunkStrings = chunkText(text);
    const chunks: { text: string; embedding: number[] }[] = [];
    for (const t of chunkStrings) {
      chunks.push({ text: t, embedding: await getEmbedding(t) });
    }

    const id = randomUUID();
    documentStore.set(id, { id, fileName, chunks, createdAt: new Date().toISOString() });

    if (documentStore.size > MAX_ACTIVE_DOCUMENTS) {
      const oldest = [...documentStore.values()].sort((a, b) => a.createdAt.localeCompare(b.createdAt))[0];
      documentStore.delete(oldest.id);
    }

    res.json({ documentId: id, fileName, chunkCount: chunks.length });
  } catch (err: any) {
    res.status(500).json({ error: `Failed to read document: ${err.message}` });
  }
});

// API: immediately frees a document's in-memory chunks/embeddings — the
// frontend's "remove attached document" affordance.
app.delete('/api/documents/:id', (req, res) => {
  if (!documentStore.has(req.params.id)) {
    return res.status(404).json({ success: false, error: "Document not found — it may already have been removed." });
  }
  documentStore.delete(req.params.id);
  res.json({ success: true });
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
  const { text, format } = req.body as { text?: string; format?: "wav" | "ogg" };
  if (!text || !text.trim()) {
    return res.status(400).json({ error: "text is required." });
  }

  try {
    const response = await fetch(`${SPEECH_SERVICE_URL}/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, ...(format ? { format } : {}) })
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Speech service returned ${response.status}: ${detail}`);
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());
    // Defaults to wav (existing web UI behavior, unchanged) — must match
    // whatever format was actually requested, since Telegram voice notes
    // need an accurate audio/ogg label, not a hardcoded audio/wav one.
    res.set('Content-Type', format === 'ogg' ? 'audio/ogg' : 'audio/wav');
    res.send(audioBuffer);
  } catch (err: any) {
    console.error("Speech synthesis failed:", err.message);

    if (!speechServiceProcess) {
      console.warn("[speech] Speech service process is down — attempting to restart it.");
      startSpeechService().catch(e => console.error("[speech] Restart attempt failed:", e));
    }

    res.status(500).json({ error: `Speech synthesis failed: ${err.message}` });
  }
});

// API: Enroll a face for facial recognition — forwards straight through to
// the vision service as JSON, same proxying convention as /api/transcribe
// and /api/speak above. No temp file involved (unlike saveBase64ImageToFile,
// used elsewhere for handing a path to file-based CLI tools) since this
// never leaves JSON/HTTP on either side.
app.post('/api/vision/enroll', async (req, res) => {
  const { name, base64, mimeType } = req.body as { name?: string; base64?: string; mimeType?: string };
  if (!name || !name.trim()) {
    return res.status(400).json({ error: "name is required." });
  }
  if (!base64) {
    return res.status(400).json({ error: "An attached photo is required to enroll a face." });
  }

  try {
    const response = await fetch(`${VISION_SERVICE_URL}/enroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, base64, mimeType: mimeType ?? "" }),
      // Face detection is CPU-bound and blocks the vision service's event
      // loop for its duration — under contention (e.g. the background scan
      // loop mid-tick) confirmed directly that this can run slow. Without a
      // timeout, a slow/stuck vision service would hang this fetch
      // indefinitely, which locks the shared requestInFlight flag on
      // /api/chat's own enrollment branch below and blocks every other
      // message until it resolves.
      signal: AbortSignal.timeout(10000)
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      return res.status(response.status).json({ error: (data as any).detail || "Face enrollment failed." });
    }

    res.json(data);
  } catch (err: any) {
    console.error("Face enrollment failed:", err.message);
    res.status(500).json({
      error: `Could not reach the vision service: ${err.message}. Is it running (VISION_SERVICE_AUTOSTART)?`
    });
  }
});

// API: Send chat prompt to the assistant
app.post('/api/chat', async (req, res) => {
  const { message, attachedImage, openModelId, currentDocumentId, channel } = req.body as {
    message: string;
    attachedImage?: { base64: string; mimeType: string } | null;
    openModelId?: string | null;
    currentDocumentId?: string | null;
    channel?: "telegram";
  };
  // Tags conversationHistory/archive entries by origin so a later recall can
  // be scoped to "on telegram" specifically (see CHANNEL_TRIGGER_PATTERN
  // below). Deliberately a plain string suffix, not a structured field, to
  // match this file's existing string[]-based history/archive format rather
  // than restructuring it.
  const channelTag = channel === "telegram" ? " (Telegram)" : "";
  const hasAttachedImage = !!attachedImage?.base64;
  // False both when nothing is attached and when the id is stale (e.g. the
  // server restarted since the frontend last uploaded a document) — either
  // way, there's no in-memory chunk/embedding state to answer from.
  const hasActiveDocument = !!currentDocumentId && documentStore.has(currentDocumentId);

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

  // Print confirm/cancel gets checked before anything else in the cascade
  // — a bare "print" or "yes" must win over every other intent (recall,
  // search, modification, etc.), the same way endSessionPattern above
  // short-circuits first. Only active while a slice is actually pending.
  const PRINT_CONFIRM_PATTERN = /^\s*(print(?:\s+it)?|yes,?\s*print(?:\s+it)?|go ahead(?:\s+and\s+print)?|start(?:\s+the)?\s+print)\s*[.!]?\s*$/i;
  const PRINT_CANCEL_PATTERN = /^\s*(no|cancel|nevermind|never\s*mind|skip|don'?t print)\s*[.!]?\s*$/i;
  if (pendingPrintJob) {
    if (Date.now() - pendingPrintJob.createdAt > PENDING_PRINT_EXPIRY_MS) {
      pendingPrintJob = null;
    } else if (PRINT_CONFIRM_PATTERN.test(message.trim())) {
      const job = pendingPrintJob;
      pendingPrintJob = null;
      requestInFlight = false;
      conversationHistory.push(`User: ${message}`);
      if (!isBambuConnectInstalled()) {
        const reply = "Bambu Connect doesn't seem to be installed — install it and sign into your Bambu account first (see PRINTER_SETUP.md), then try again.";
        conversationHistory.push(`Assistant: ${reply}`);
        saveHistory();
        return res.json({ response: reply, hasProposedChanges: false });
      }
      const launchResult = launchBambuConnect(job.gcodePath, job.subject);
      const reply = launchResult.success
        ? `Handed "${job.subject}" off to Bambu Connect — check its window to confirm and start the print.`
        : `Couldn't hand the file off to Bambu Connect: ${launchResult.error}`;
      conversationHistory.push(`Assistant: ${reply}`);
      saveHistory();
      return res.json({ response: reply, hasProposedChanges: false });
    } else if (PRINT_CANCEL_PATTERN.test(message.trim())) {
      try { fs.unlinkSync(pendingPrintJob.gcodePath); } catch { /* already gone or never existed */ }
      pendingPrintJob = null;
      requestInFlight = false;
      conversationHistory.push(`User: ${message}`);
      const reply = "Cancelled — the print job wasn't started.";
      conversationHistory.push(`Assistant: ${reply}`);
      saveHistory();
      return res.json({ response: reply, hasProposedChanges: false });
    }
    // Anything else falls through to normal routing below — a pending
    // print survives unrelated small talk rather than being force-consumed.
  }

  conversationHistory.push(`User${channelTag}: ${message}`);
  saveHistory();

  const personality = loadFile(PERSONALITY_FILE);
  const memory = loadMemory();

  // "make it bigger", "mirror it", "simplify it", "fill the holes" all
  // contain words (edit/change/update) that wantsModification's own regex
  // below would otherwise swallow, misrouting a mesh-edit request into the
  // self-modification (personality.txt/memory.json patch) flow — confirmed
  // directly. Computed first so wantsModification can exclude it, same
  // "gate against something with a broader net" pattern as
  // NEGATED_MODIFICATION_PATTERN just below.
  const MESH_SCALE_PATTERN = /\b(make it|scale it|resize it|scale the (model|mesh)|resize the (model|mesh))\s+(\d+(\.\d+)?\s*(x|times)|(a\s+)?(bit\s+)?(bigger|larger|smaller|(twice|half)( as (big|large))?)|by\s+\d+%)\b/i;
  const MESH_MIRROR_PATTERN = /\b(mirror|flip)\s+(it|the (model|mesh))\b/i;
  const MESH_SIMPLIFY_PATTERN = /\b(simplify|decimate|reduce the (poly|polygon|face)( ?count)?|lower the (poly|polygon|detail)( ?count| level)?|fewer (polygons|faces|triangles))\b.{0,20}\b(it|the (model|mesh))?\b/i;
  const MESH_FILL_HOLES_PATTERN = /\b(fill|patch|close)\s+(the\s+)?(holes?|gaps?)\b/i;
  const looksLikeMeshEditRequest = (
    MESH_SCALE_PATTERN.test(message) ||
    MESH_MIRROR_PATTERN.test(message) ||
    MESH_SIMPLIFY_PATTERN.test(message) ||
    MESH_FILL_HOLES_PATTERN.test(message)
  );
  console.log("LOOKS LIKE MESH EDIT REQUEST:", looksLikeMeshEditRequest);

  // "Remember this face as Noel" contains "remember," which
  // wantsModification's own regex below would otherwise swallow, misrouting
  // face enrollment into the self-modification (personality.txt/memory.json
  // patch) flow — same collision, same fix pattern as looksLikeMeshEditRequest
  // above. Gated on hasAttachedImage since enrollment is meaningless without
  // a photo to enroll.
  // Deliberately loose on the trigger (just "remember [this/that/him/her/
  // them] ... face/person" somewhere nearby) — the actual name is pulled
  // out separately by extractEnrollName, which tries several natural
  // phrasings ("as X", "name is X", "this is X", "called X"). Confirmed
  // live: requiring the name to immediately follow "as" missed "remember
  // this face, name is Joshua" entirely, silently falling through to plain
  // chat instead of enrolling anything.
  const ENROLL_FACE_TRIGGER_PATTERN = /\bremember (?:this|that|him|her|them)\b[\s\S]{0,20}\b(?:face|person)\b/i;
  const faceEnrollName = hasAttachedImage && ENROLL_FACE_TRIGGER_PATTERN.test(message) ? extractEnrollName(message) : null;
  const looksLikeFaceEnrollRequest = faceEnrollName !== null;
  console.log("LOOKS LIKE FACE ENROLL REQUEST:", looksLikeFaceEnrollRequest, "| name:", faceEnrollName);

  // Model comparison ("compare qwen3.5:4b and qwen3.5:9b on: write a haiku").
  // Requires BOTH the word "compare"/"vs"/"versus" AND at least 2 colon-form
  // Ollama model identifiers (e.g. "qwen3.5:9b") — that combination is
  // distinctive enough (nothing else in this cascade produces or expects
  // colon-form tokens) that it's checked and branched on first, ahead of
  // every other early-return intent, rather than threaded through each of
  // their exclusion chains like looksLikeHardwareQuery/looksLikeModelRecommendQuery
  // below had to be. The leading-letter requirement is deliberate — confirmed
  // live that a plain "compare 10:30 and 11:45" false-positived on a naive
  // \w+:\w+ pattern (both look like colon-form tokens); every real Ollama
  // model name starts with a letter (qwen3.5, llama3.1, gemma4, ...), so this
  // excludes bare numeric/time-like tokens without excluding any real model.
  const MODEL_NAME_TOKEN_PATTERN = /\b[a-zA-Z][\w.-]*:[\w.-]+\b/g;
  const COMPARE_MODELS_TRIGGER_PATTERN = /\b(compare|vs\.?|versus)\b/i;
  const compareModelTokens = message.match(MODEL_NAME_TOKEN_PATTERN) ?? [];
  const looksLikeCompareModelsRequest = COMPARE_MODELS_TRIGGER_PATTERN.test(message) && compareModelTokens.length >= 2;
  console.log("LOOKS LIKE COMPARE MODELS REQUEST:", looksLikeCompareModelsRequest, "| tokens:", compareModelTokens);

  // "No changes to your personality" / "I didn't change anything" mention the
  // keyword while explicitly saying nothing should happen — without this,
  // they trip wantsModification just as wrongly as "do you remember" tripped
  // it on the word "remember" (see REMEMBER_QUESTION_PATTERN above).
  const NEGATED_MODIFICATION_PATTERN = /\b(no|not|don'?t|didn'?t|without|never)\s+(\w+\s+){0,2}(changes?|modif(y|ication)|updates?)\b/i;

  const wantsModification = !looksLikeMeshEditRequest && !looksLikeFaceEnrollRequest && !looksLikeCompareModelsRequest && !NEGATED_MODIFICATION_PATTERN.test(message) && (
    (REMEMBER_QUESTION_PATTERN.test(message) || FORGETFUL_STATEMENT_PATTERN.test(message))
      ? /(modify|change|rewrite|update|edit|improve|refactor|memorize|store|save)/i.test(message)
      : /(modify|change|rewrite|update|edit|improve|refactor|remember|memorize|store|save)/i.test(message)
  );

  console.log( "WANTS MODIFICATION:", wantsModification );

  const recallDate = extractDateFromMessage(message);
  const mentionsRecall = RECALL_TRIGGER_PATTERN.test(message);
  const isRecallQuery = !wantsModification && (
    mentionsRecall || (recallDate !== null && /what|talk|discuss|speak|say/i.test(message))
  );

  // "What did we talk about on Telegram" narrows a recall (date- or
  // topic-based) to only channel-tagged messages — see channelTag in
  // /api/chat for how the tag gets into archived lines in the first place.
  // A bare topic/date recall with no channel mention stays untouched and
  // searches everything, by design (Noel: topic recall should look at all
  // history unless a channel or date is also given).
  const CHANNEL_TRIGGER_PATTERN = /\b(on telegram|via telegram|in telegram|telegram (conversation|chat|message))\b/i;
  const recallChannel: "telegram" | null = CHANNEL_TRIGGER_PATTERN.test(message) ? "telegram" : null;

  // A follow-up on an already-answered recall query (e.g. "give me specifics", "quote it") — doesn't re-trigger isRecallQuery on its own, but should still get the archived context and the smarter model.
  // Same exclusions as isStickySearchFollowup below and for the same
  // confirmed-live reason — an unrelated, clearly-distinct request landing
  // during this sticky window shouldn't get swallowed by it either.
  const isStickyRecallFollowup = !wantsModification && !isRecallQuery && !looksLikeMeshEditRequest && !looksLikeFaceEnrollRequest && !FUSION_TRIGGER_PATTERN.test(message) && stickyRecallTurnsRemaining > 0;

  // "What does core integrity mean" is answered fine from the general explanation now in personality.txt — but "why isn't it 100%" or "what's down" needs the actual live health state, which the model has no way to
  // know on its own. This always fetches fresh (no sticky reuse): unlike a search result, service health can flip within seconds, so reusing stale
  // status from a previous turn would risk telling the user a service is down when it's since recovered, or vice versa.
  const STATUS_TRIGGER_PATTERN = /\b(core integrity|neural load|model temp|diverg(ence)?[\s_-]?buf(fer)?|system status|what'?s down|which service|is (everything|anything) (down|broken|working|up)|are you (down|working|online)|health check)\b/i;
  const looksLikeStatusQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && STATUS_TRIGGER_PATTERN.test(message);

  // "Who showed up" / "who's there" needs the vision service's actual
  // current camera state, not a guess — without this, the model has nothing
  // grounded to answer from beyond a proactive greeting that may already be
  // out of its context window, and (confirmed directly) will fabricate an
  // answer rather than say it doesn't know. Same "fetch real data, inject it,
  // let the model phrase the answer" pattern as looksLikeStatusQuery above.
  const VISION_STATUS_TRIGGER_PATTERN = /\b(who('?s| is| was) (there|here|around|present|watching)|who (just )?(showed up|showing up|walked in|appeared|arrived)|who did you see|is (anyone|someone) (there|here)|who'?s in (the (room|frame|view)))\b/i;
  const looksLikeVisionStatusQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && VISION_STATUS_TRIGGER_PATTERN.test(message);

  // "What's my GPU" / "how much VRAM do I have" is a factual question about
  // the user's own machine, not about Noah's own operational health —
  // deliberately its own intent rather than folded into
  // looksLikeStatusQuery, which is framed around service uptime ("Core
  // Integrity: X%") and would otherwise need to grow broader in a way that
  // risks exactly the kind of cross-trigger collision already found once
  // this session (the sticky-search-followup bug).
  const HARDWARE_TRIGGER_PATTERN = /\b(what'?s my (gpu|hardware|vram)|how much vram|what gpu (do i have|am i running)|my (graphics|video) card|check my hardware)\b/i;
  const looksLikeHardwareQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && HARDWARE_TRIGGER_PATTERN.test(message);

  // "What models can I run" — combines real data (already-pulled models,
  // via listPulledModels' actual file sizes) with OLLAMA_MODEL_VRAM_TABLE's
  // necessarily-approximate suggestions for what else would fit. Same
  // "own intent, not folded into status" reasoning as hardware above.
  const MODEL_RECOMMEND_TRIGGER_PATTERN = /\b(what models? (can|could|should) i run|recommend(ed)? (a |an )?(ollama )?model|which models? (fit|would fit)|models? for my (gpu|hardware|vram))\b/i;
  const looksLikeModelRecommendQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && MODEL_RECOMMEND_TRIGGER_PATTERN.test(message);

  // "Give me today's headlines" is a news-digest request, not a factual
  // lookup — it gets its own trigger and its own fetch path (SearXNG's news
  // category + day filter) rather than a plain web search, which just
  // surfaces static homepages for a bare term like "headlines".
  const HEADLINES_TRIGGER_PATTERN = /\b(headlines|breaking news|today'?s news|news today|on the news|what'?s (been )?reported|happening in the world)\b/i;
  const looksLikeHeadlinesQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && !looksLikeModelRecommendQuery && HEADLINES_TRIGGER_PATTERN.test(message);

  const SEARCH_TRIGGER_PATTERN = /\b(latest news|who won|release date|launch date|premiere date|when('s| is| does| will).{0,30}(come out|coming out|releas(e|ing)|drop(ping)?|launch(ing)?|premier(e|ing))|current (weather|price|score|exchange rate)|how much (is|does|would)|price of|what'?s the weather|weather (today|forecast|right now))\b/i;
  const looksLikeSearchQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && !looksLikeModelRecommendQuery && !looksLikeHeadlinesQuery && SEARCH_TRIGGER_PATTERN.test(message);

  // A follow-up on an already-answered search or headlines request (e.g.
  // "give me the link to the sites you used", "which one said that") —
  // needs the same results (with real URLs) rather than the model
  // improvising from its own summary. Confirmed live (on master): without
  // the explicit mesh-edit/face-enroll/fusion exclusions below, an
  // unrelated request landing right after a search/headlines exchange got
  // swallowed into this sticky-followup path instead of ever reaching its
  // own, much more specific trigger pattern. The first two flags are
  // already computed by this point and can be referenced directly;
  // FUSION_TRIGGER_PATTERN is tested directly (not via
  // looksLikeFusionRequest, which isn't computed yet) for the same
  // TDZ-ordering reason it's now a top-level const.
  const isStickySearchFollowup = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && !looksLikeModelRecommendQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !looksLikeMeshEditRequest && !looksLikeFaceEnrollRequest && !FUSION_TRIGGER_PATTERN.test(message) && stickySearchTurnsRemaining > 0;

  const looksLikeFusionRequest = !looksLikeMeshEditRequest && !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && !looksLikeModelRecommendQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !isStickySearchFollowup && FUSION_TRIGGER_PATTERN.test(message);

  // A Fusion request either describes explicit parametric geometry (shape
  // words, dimensions) — handled by the existing code-generation path — or
  // names a real-world subject with none of that vocabulary (e.g. "a 3d
  // model of spiderman"), which the model can't write CAD code for at all.
  // Those instead go through the image-to-3D pipeline: find a reference
  // photo, run it through a local image-to-mesh model, import the result.
  // Removal requests always stay on the code path (there's no image
  // involved in deleting existing bodies).
  const REMOVAL_VERB_PATTERN = /\b(remove|delete|clear)\b/i;
  // Deliberately excludes bare dimension mentions (e.g. "5cm") — a named
  // subject like "a fox, make it 5cm tall" should still go through the
  // image pipeline; only an actual shape word means parametric CAD code.
  const SHAPE_WORD_PATTERN = /\b(cylinder|box|cube|sphere|cone|rectangle|pyramid|prism|torus|ring|tube|wedge|radius|diameter|extrude|revolve|fillet|chamfer)\b/i;
  const looksLikeImageFusionRequest = looksLikeFusionRequest && !REMOVAL_VERB_PATTERN.test(message) && !SHAPE_WORD_PATTERN.test(message);
  const looksLikeParametricFusionRequest = looksLikeFusionRequest && !looksLikeImageFusionRequest;

  // Ollama's "thinking" mode measurably improves how carefully the model reasons through a request, but costs ~10-15+ seconds of extra latency
  // per response (tested directly: a trivial one-sentence question took
  // 15.5s with it on vs ~1-2s off) — far too slow to enable by default, but worth it when the user explicitly asks for more thoroughness.
  const DIG_DEEPER_PATTERN = /\b(dig deeper|dig into (that|this|it)|look into (that|this|it) more|go deeper|expand on (that|this|it)|elaborate( on (that|this|it))?)\b/i;
  const wantsDeeperThinking = DIG_DEEPER_PATTERN.test(message);

  console.log("IS RECALL QUERY:", isRecallQuery);
  console.log("IS STICKY RECALL FOLLOWUP:", isStickyRecallFollowup, "| turns remaining:", stickyRecallTurnsRemaining);
  console.log("LOOKS LIKE STATUS QUERY:", looksLikeStatusQuery);
  console.log("LOOKS LIKE VISION STATUS QUERY:", looksLikeVisionStatusQuery);
  console.log("LOOKS LIKE HARDWARE QUERY:", looksLikeHardwareQuery);
  console.log("LOOKS LIKE MODEL RECOMMEND QUERY:", looksLikeModelRecommendQuery);
  console.log("LOOKS LIKE HEADLINES QUERY:", looksLikeHeadlinesQuery);
  console.log("LOOKS LIKE SEARCH QUERY:", looksLikeSearchQuery);
  console.log("IS STICKY SEARCH FOLLOWUP:", isStickySearchFollowup, "| turns remaining:", stickySearchTurnsRemaining);
  console.log("LOOKS LIKE FUSION REQUEST:", looksLikeFusionRequest, "| parametric:", looksLikeParametricFusionRequest, "| image-based:", looksLikeImageFusionRequest);
  console.log("WANTS DEEPER THINKING:", wantsDeeperThinking);

  // Short messages like ("hi", "thanks", "ok") are never complex on keyword grounds alone —
  // skip straight to the fast model rather than utilize the smart model to decrease latency
  // Modification requests still go to the smart model regardless of length, since
  // they need to reliably produce structured JSON output.
  const wordCount = message.trim().split(/\s+/).length;

  // Same negation chain isPlainChatTurn uses further down (it can't be
  // referenced directly here — isComplex, which needs this, is computed
  // before isPlainChatTurn exists) — a document question is exactly a
  // plain chat turn that also happens to have a document active, so it
  // sits at the bottom of the same priority cascade and can't collide
  // with anything above it.
  const looksLikeDocumentQuestion = hasActiveDocument && !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && !looksLikeModelRecommendQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !isStickySearchFollowup && !looksLikeFusionRequest;
  console.log("LOOKS LIKE DOCUMENT QUESTION:", looksLikeDocumentQuestion);

  const isComplex = wantsModification || isRecallQuery || isStickyRecallFollowup || looksLikeStatusQuery || looksLikeVisionStatusQuery || looksLikeHardwareQuery || looksLikeModelRecommendQuery || looksLikeHeadlinesQuery || looksLikeSearchQuery || isStickySearchFollowup || looksLikeFusionRequest || looksLikeMeshEditRequest || looksLikeDocumentQuestion || (
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
  // conversations) and their "I have no record of that day" replies would
  // otherwise false-positive as uncertainty below — the search fallback only
  // makes sense for genuine open-ended chat turns, not those paths.
  const isPlainChatTurn = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeVisionStatusQuery && !looksLikeHardwareQuery && !looksLikeModelRecommendQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !isStickySearchFollowup && !looksLikeFusionRequest;

  const searchInstructions = isPlainChatTurn ? `
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

    // Was 90s — too tight for Fusion code generation specifically, which can
    // legitimately take well over a minute if the model rambles through a
    // harder multi-step request (observed directly: sometimes 6-7s, sometimes
    // much longer for the exact same prompt, since temperature>0 means it
    // isn't deterministic). A slow-but-correct response beats a hard abort
    // that forces a retry from scratch. This timeout is shared by every
    // request type, not just Fusion ones, so a truly stuck Ollama call now
    // takes longer to surface as an error too — an acceptable tradeoff here.
    const timeout = setTimeout(() => {
      controller.abort();
    }, 180000);

    async function callOllama(prompt: string, numPredict: number, think: boolean = false, images?: string[], model: string = selectedModel, signal: AbortSignal = controller.signal): Promise<string> {
      const res = await fetch(OLLAMA_URL, {
        method: "POST",
        signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
          think,
          // Ollama's vision models take images as a flat array of raw
          // base64 strings (no data: URI prefix) alongside the prompt.
          ...(images && images.length > 0 ? { images } : {}),
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
      if (!res.ok) throw new Error(`Ollama connection failed: HTTP ${res.status} ${await res.text().catch(() => "")}`.trim());
      const data = await res.json() as { response: string };
      return data.response;
    }

    // Model comparison is checked before every other early-return
    // path — its trigger (the word "compare"/"vs"/"versus" AND 2+
    // colon-form model tokens like "qwen3.5:9b") is distinctive enough
    // that it doesn't need to compete for priority with anything else in
    // this cascade, so it gets first refusal unconditionally.
    if (looksLikeCompareModelsRequest) {
      const [modelA, modelB] = compareModelTokens.slice(0, 2);
      const ignoredModelNote = compareModelTokens.length > 2
        ? ` (ignored the rest: ${compareModelTokens.slice(2).join(", ")} — comparisons are limited to 2 models at a time)`
        : "";

      const pulled = await listPulledModels();
      const pulledNames = new Set(pulled.map(m => m.name));
      const missing = [modelA, modelB].filter(m => !pulledNames.has(m));

      if (missing.length > 0) {
        const availableList = pulled.length > 0 ? pulled.map(m => m.name).join(", ") : "(none pulled)";
        const compareReplyText = `I don't have ${missing.join(" and ")} pulled, so I can't run that comparison. Models available locally: ${availableList}.`;
        conversationHistory.push(`Assistant: ${compareReplyText}`);
        saveHistory();
        clearTimeout(timeout);
        return res.json({ response: compareReplyText, hasProposedChanges: false });
      }

      const promptExtractionPrompt = `Extract just the actual question or task the user wants both models to answer, stripped of any "compare X and Y" framing. Reply with ONLY that question/task text and nothing else.\n\nRequest: "${message}"`;
      const rawComparePrompt = await callOllama(promptExtractionPrompt, 40);
      const comparePrompt = rawComparePrompt.replace(/["'\n]+/g, " ").trim() || message;
      console.log("EXTRACTED COMPARISON PROMPT:", comparePrompt);

      const comparison = await runModelComparison(modelA, modelB, comparePrompt);
      const compareReplyText = `Here's how ${modelA} and ${modelB} answered "${comparePrompt}"${ignoredModelNote}:`;

      conversationHistory.push(`Assistant: ${compareReplyText}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({ response: compareReplyText, comparison, hasProposedChanges: false });
    }

    // Face enrollment is its own early-return path, checked before the
    // Fusion/pose-reference/general-vision-query attached-image branches
    // below — "remember this face as X" is a much more specific phrasing
    // than a plain object photo or a "reconstruct this as 3D" request, so it
    // needs first refusal on any attached image, same "most specific wins
    // first" ordering already used for pose-reference disambiguation.
    if (looksLikeFaceEnrollRequest) {
      const enrollName = faceEnrollName!;
      let enrollReplyText: string;
      try {
        const enrollRes = await fetch(`${VISION_SERVICE_URL}/enroll`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: enrollName, base64: attachedImage!.base64, mimeType: attachedImage!.mimeType }),
          // See the /api/vision/enroll proxy above — same reasoning, but
          // more important here specifically: this fetch runs inside the
          // /api/chat handler under the shared requestInFlight lock, so a
          // hung vision service would block every other chat message too.
          signal: AbortSignal.timeout(10000)
        });
        const enrollData = await enrollRes.json().catch(() => ({})) as { success?: boolean; detail?: string };
        enrollReplyText = enrollRes.ok
          ? `Got it — I'll remember ${enrollName}'s face.`
          : `Couldn't enroll that face: ${enrollData.detail || "unknown error"}`;
      } catch (err: any) {
        enrollReplyText = `Couldn't reach the vision service to enroll that face: ${err.message}. Is it running (VISION_SERVICE_AUTOSTART)?`;
      }

      conversationHistory.push(`Assistant: ${enrollReplyText}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({ response: enrollReplyText, hasProposedChanges: false });
    }

    // Fusion 360 model generation is handled as its own early-return path —
    // it needs a narrowly-scoped code-generation prompt (not the general
    // chat/personality prompt), a real side effect (executing in a running
    // Fusion session via the bridge add-in), and a result-shaped response —
    // none of which fit the JSON-modification/recall/search pipeline below.
    if (looksLikeParametricFusionRequest) {
      const fusionPrompt = `You are generating Python code for Autodesk Fusion 360's API to create 3D geometry. This code will be executed via exec() inside a running Fusion 360 session. These variables already exist in scope — do not import adsk or redefine them:
- adsk (the adsk module itself)
- app (adsk.core.Application instance)
- ui (the active UserInterface)
- design (the active adsk.fusion.Design)
- rootComp (design.rootComponent)

Rules:
- CRITICAL: output ONLY one \`\`\`python code block and NOTHING else — no explanation, no reasoning, and NO comment lines (lines starting with #) anywhere in the code. If you're unsure of the right approach, pick your best option silently and just output working code for it — never think out loud inside the code block, that wastes your output budget and can cut off the closing \`\`\` before it's reached.
- Create sketches with rootComp.sketches.add(rootComp.xYConstructionPlane) (or another construction plane).
- Circles: sketch.sketchCurves.sketchCircles.addByCenterRadius(point, radius).
- Rectangles: sketch.sketchCurves.sketchLines.addTwoPointRectangle(p1, p2) — note the .sketchLines in between, addTwoPointRectangle is NOT a direct member of sketchCurves.
- Extrude with rootComp.features.extrudeFeatures. Revolve with rootComp.features.revolveFeatures.
- adsk.core.Point3D.create ALWAYS requires exactly 3 arguments (x, y, z) — e.g. Point3D.create(0, 0, 0), never Point3D.create(0, 0).
- p1 and p2 for addTwoPointRectangle MUST be diagonally opposite corners — they must differ in BOTH x and y. If they share the same x or the same y, that's a zero-area degenerate rectangle and Fusion will reject it with "Invalid input points". E.g. for a 5x6 rectangle: p1=(0,0,0), p2=(5,6,0) — never p1=(2.5,-4.8,0), p2=(-2.5,-4.8,0) (same y on both).
- Double-check that rectangle/shape coordinates actually produce the exact dimensions requested (e.g. for a rectangle addTwoPointRectangle(p1, p2), the distance between p1 and p2 on each axis must equal the requested width/length).
- Always call .add(input) on the SPECIFIC feature collection you got createInput from (e.g. extrudes.add(...), revolves.add(...)) — never rootComp.features.add(...), which does not exist and will fail with AttributeError.
- Keep it to simple, valid parametric geometry — do not guess at API members you're not sure exist.

Example 1, for "a cylinder 5cm radius and 10cm tall":
\`\`\`python
sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
sketch.sketchCurves.sketchCircles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), 5)
prof = sketch.profiles.item(0)
extrudes = rootComp.features.extrudeFeatures
extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
extInput.setDistanceExtent(False, adsk.core.ValueInput.createByReal(10))
extrudes.add(extInput)
\`\`\`

Example 2, for "a box 4cm wide, 6cm long, and 3cm tall":
\`\`\`python
sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
p1 = adsk.core.Point3D.create(0, 0, 0)
p2 = adsk.core.Point3D.create(4, 6, 0)
sketch.sketchCurves.sketchLines.addTwoPointRectangle(p1, p2)
prof = sketch.profiles.item(0)
extrudes = rootComp.features.extrudeFeatures
extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
extInput.setDistanceExtent(False, adsk.core.ValueInput.createByReal(3))
extrudes.add(extInput)
\`\`\`

Example 3, for "a sphere with 4cm radius" — there's no sphere primitive, so sketch a semicircle profile (an arc plus a straight diameter line closing it) and revolve it 360 degrees around the diameter line:
\`\`\`python
sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
centerPoint = adsk.core.Point3D.create(0, 0, 0)
startPoint = adsk.core.Point3D.create(0, 4, 0)
endPoint = adsk.core.Point3D.create(0, -4, 0)
sketch.sketchCurves.sketchArcs.addByCenterStartSweep(centerPoint, startPoint, math.pi)
sketch.sketchCurves.sketchLines.addByTwoPoints(startPoint, endPoint)
prof = sketch.profiles.item(0)
revolves = rootComp.features.revolveFeatures
revInput = revolves.createInput(prof, rootComp.yConstructionAxis, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
revInput.setAngleExtent(False, adsk.core.ValueInput.createByString("360 deg"))
revolves.add(revInput)
\`\`\`

Example 4, for "a cylinder with a hole through the center" — draw both circles in the SAME sketch and extrude the ring-shaped profile directly (the one with 2 profile loops, not 1) as a single new body. Do not use a separate cut/subtract step for this — one extrude of the ring profile already produces a hollow tube:
\`\`\`python
sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
centerPoint = adsk.core.Point3D.create(0, 0, 0)
sketch.sketchCurves.sketchCircles.addByCenterRadius(centerPoint, 5)
sketch.sketchCurves.sketchCircles.addByCenterRadius(centerPoint, 2)
ringProfile = None
for p in sketch.profiles:
    if p.profileLoops.count == 2:
        ringProfile = p
extrudes = rootComp.features.extrudeFeatures
extInput = extrudes.createInput(ringProfile, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
extInput.setDistanceExtent(False, adsk.core.ValueInput.createByReal(10))
extrudes.add(extInput)
\`\`\`

math is already available (e.g. math.pi) — do not import it.

If the request is to remove/delete/clear existing geometry instead of creating something, delete all bodies and sketches in the root component:
\`\`\`python
for body in list(rootComp.bRepBodies):
    body.deleteMe()
for sketch in list(rootComp.sketches):
    sketch.deleteMe()
\`\`\`

Now generate code for this request: "${message}"`;

      const fusionResponse = await callOllama(fusionPrompt, 2500);
      const codeMatch = fusionResponse.match(/```(?:python)?\s*\n([\s\S]*?)```/);

      let fusionReplyText: string;

      if (!codeMatch) {
        fusionReplyText = "I couldn't produce a valid code block for that model request. Try rephrasing it as a simpler shape.";
      } else {
        const generatedCode = codeMatch[1].trim();
        const { available: bridgeAvailable, justLaunched } = await ensureFusionBridgeAvailable();

        if (!bridgeAvailable) {
          fusionReplyText = justLaunched
            ? `I started Fusion 360 for you, but it's still loading (or the NoahFusionBridge add-in isn't set to run on startup) — give it a bit and try again. To make this automatic, check "Run on Startup" for NoahFusionBridge in Fusion's Add-Ins panel.\n\nGenerated code:\n${generatedCode}`
            : `I generated code for this, but couldn't reach the Fusion 360 bridge at ${FUSION_BRIDGE_URL}, and couldn't find Fusion 360 to launch it automatically — make sure it's installed and running with the NoahFusionBridge add-in active.\n\nGenerated code:\n${generatedCode}`;
        } else {
          const result = await executeFusionScript(generatedCode);
          fusionReplyText = result.success
            ? `Done — applied that in Fusion 360.\n\nCode used:\n${generatedCode}`
            : `Fusion rejected the generated code:\n${result.error}\n\nCode attempted:\n${generatedCode}`;
        }
      }

      conversationHistory.push(`Assistant: ${fusionReplyText}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({ response: fusionReplyText, hasProposedChanges: false });
    }

    // Image-to-3D path: the request names a real subject (not parametric
    // geometry), so there's no CAD code to generate — instead find a
    // reference photo, run it through the local Hunyuan3D-2 model to get a
    // mesh, then import that mesh directly into the Fusion document.
    if (looksLikeImageFusionRequest) {
      const subject = extractFusionSubject(message);
      const targetSizeCm = extractTargetSizeCm(message);
      const userSuppliedImageUrl = extractImageUrl(message);
      let imageReplyText: string = "Something went wrong generating that model.";
      let generatedModel: GeneratedModelEntry | null = null;

      const { available: bridgeAvailable, justLaunched } = await ensureFusionBridgeAvailable();
      if (!bridgeAvailable) {
        imageReplyText = justLaunched
          ? `I started Fusion 360 for you, but it's still loading (or the NoahFusionBridge add-in isn't set to run on startup) — give it a bit and try again. To make this automatic, check "Run on Startup" for NoahFusionBridge in Fusion's Add-Ins panel.`
          : `I'd need Fusion 360 running to import a model of "${subject}", and couldn't find it installed to launch automatically — make sure it's running with the NoahFusionBridge add-in active.`;
      } else {
        let imagePath: string | null = null;
        let multiviewPaths: { front: string; left: string; back: string } | null = null;
        let referenceLabel = "";
        const tempFiles: string[] = [];

        // An attached image is the most direct signal available — skip the
        // URL/search steps entirely and use exactly what was dropped in.
        if (hasAttachedImage) {
          imagePath = saveBase64ImageToFile(attachedImage!.base64, attachedImage!.mimeType);
          referenceLabel = "the image you attached";
          tempFiles.push(imagePath);
        } else if (userSuppliedImageUrl) {
          const downloadPath = path.join(IMAGE23D_TMP_DIR, `${Date.now()}.jpg`);
          const downloaded = await downloadImageToFile(userSuppliedImageUrl, downloadPath);
          if (downloaded) {
            imagePath = downloadPath;
            referenceLabel = `this reference image: ${userSuppliedImageUrl}`;
            tempFiles.push(imagePath);
          } else {
            imageReplyText = `Found the image link you gave me for "${subject}" but couldn't download it to work with.`;
          }
        } else {
          // The multiview search still applies to a partial-subject request
          // (a bust genuinely does have a front/side/back worth capturing
          // separately) — what has to change is what "correct framing"
          // means. Requiring full-body for "a headbust of spiderman" would
          // reject every real candidate, since a bust or close-up is BY
          // DEFINITION not full-body. The verification prompt below keys
          // off this instead of hardcoding full-body — and uses subject-
          // agnostic language ("the entire X") rather than "full-body" /
          // "face or head", which are person/creature-specific and don't
          // fit a request like "a model of a car".
          const wantsPartialSubject = /\b(headbust|bust|headshot|head[\s-]?only|close-?up|closeup|portrait)\b/i.test(subject);

          // Explicit opt-out — an attached image or a pasted URL already
          // forces the single-image path above regardless of wording, but
          // there was no way to say "just use one image" in plain text when
          // relying on search (confirmed directly: it silently attempted
          // multiview anyway). Checked against the whole message, not just
          // the extracted subject, since this is a request-shaped
          // instruction rather than a description of what the subject is.
          const wantsSingleImageOnly = /\b(one|single)\s+(image|photo|picture|reference)\b|\b(no|not|skip|without|don'?t)\b[^.?!]*\bmultiview\b/i.test(message);

          if (!wantsSingleImageOnly) {
          // Experimental: try to source three angle-matched photos so
          // Hunyuan3D-2mv's real multiview conditioning can be used instead
          // of one photo — see generateMeshWithHunyuan3DMultiview's comment
          // for why three independently-searched images aren't guaranteed to
          // actually agree with each other. Only commits to this path if all
          // three searches return something; otherwise falls back to the
          // proven single-image search below.

          // Confirmed directly: without this, "spiderman front/side/back
          // view" search results included at least one image that wasn't
          // Spiderman at all, not just a different art style or pose — a
          // mismatch real enough to sink the mesh, not just a quality risk.
          // Ask the same local vision model used for image-description to
          // reject candidates before they're ever fed to the mesh pipeline.
          // Fails closed (treats an errored check as "not a match") since a
          // silently-accepted bad image is the exact failure this exists to
          // catch, and there are up to 5 other candidates left to try.
          const verifyImageMatchesSubject = async (imagePath: string): Promise<boolean> => {
            try {
              const base64 = fs.readFileSync(imagePath).toString("base64");
              // Uses its own timeout rather than the request-wide controller
              // above — confirmed directly: that controller fires once at
              // 180s for the whole /api/chat request, and up to 5 candidates
              // per angle across 3 angles can plausibly take longer than
              // that combined. Once it fires it's permanently tripped, so
              // every verification call still in flight — for every
              // angle, not just the slow one — would immediately fail and
              // get treated as "not a match," collapsing the whole multiview
              // attempt regardless of whether the images were actually fine.
              // 200 tokens of budget covers this model's occasional (short)
              // reasoning aside even with think=false, which does suppress
              // the full chain-of-thought (confirmed directly).
              // Confirmed directly: subject-match alone isn't enough — a
              // genuine close-up crop and a genuine full view both pass
              // "does this show spiderman"/"does this show a car", but
              // they're incompatible views for one mesh (Hunyuan3D-2mv's
              // own example images are all consistent full-subject
              // framing). Reject close-ups too — unless the request itself
              // wants a partial subject (a bust, a headshot), in which case
              // a close-up IS the correct framing and requiring the whole
              // subject would reject everything.
              const answer = await callOllama(
                wantsPartialSubject
                  ? `Answer with exactly one word, YES or NO: does this image clearly show ${subject}?`
                  : `Answer with exactly one word, YES or NO: does this image clearly show the entire ${subject}, not just a close-up of one part?`,
                200,
                false,
                [base64],
                CODE_MODEL,
                AbortSignal.timeout(60000)
              );
              return /\byes\b/i.test(answer);
            } catch (err) {
              console.error("Image verification failed:", err);
              return false;
            }
          };

          // Fails closed like verifyImageMatchesSubject, same reasoning —
          // an unreliable check should lose a mesh's worth of detail to the
          // single-image fallback, not risk letting a genuinely conflicting
          // set through. Ollama's images array can hold more than one
          // image per call, and a vision-capable model can reason about all
          // of them together in one response — this asks it to do exactly
          // that instead of judging each photo alone.
          const verifyImagesAreConsistentSet = async (frontPath: string, leftPath: string, backPath: string): Promise<boolean> => {
            try {
              const images = [frontPath, leftPath, backPath].map(p => fs.readFileSync(p).toString("base64"));
              const answer = await callOllama(
                `These three images are meant to be used as front, side, and back reference photos of the same subject for a 3D model. Answer with exactly one word, YES or NO: do all three actually show the same pose, art style, and appearance — not just the same subject, but consistent enough with each other to be usable as one coherent set?`,
                300,
                false,
                images,
                CODE_MODEL,
                AbortSignal.timeout(90000)
              );
              return /\byes\b/i.test(answer);
            } catch (err) {
              console.error("Cross-image consistency check failed:", err);
              return false;
            }
          };

          const ts = Date.now();
          const [frontUrls, leftUrls, backUrls] = await Promise.all([
            webImageSearch(`${subject} front view`),
            webImageSearch(`${subject} side view`),
            webImageSearch(`${subject} back view`)
          ]);
          if (frontUrls.length > 0 && leftUrls.length > 0 && backUrls.length > 0) {
            const frontPath = path.join(IMAGE23D_TMP_DIR, `${ts}_front.jpg`);
            const leftPath = path.join(IMAGE23D_TMP_DIR, `${ts}_left.jpg`);
            const backPath = path.join(IMAGE23D_TMP_DIR, `${ts}_back.jpg`);
            // Sequential, not Promise.all — confirmed directly: three
            // concurrent verification calls raced to cold-load the same
            // model into VRAM simultaneously and Ollama errored out on all
            // three. Once the model is loaded this would just queue safely,
            // but there's no cheap way to know "already loaded" from here,
            // so always serializing is the reliable option.
            const frontUsed = await downloadFirstAvailableImage(frontUrls, frontPath, verifyImageMatchesSubject);
            const leftUsed = await downloadFirstAvailableImage(leftUrls, leftPath, verifyImageMatchesSubject);
            const backUsed = await downloadFirstAvailableImage(backUrls, backPath, verifyImageMatchesSubject);
            tempFiles.push(frontPath, leftPath, backPath);
            // Per-image verification above only checks each photo in
            // isolation (right subject, right framing) — it can't catch the
            // three of them disagreeing with EACH OTHER (confirmed directly:
            // a Spiderman multiview mesh came out with two extra detached
            // legs, almost certainly because the three source photos weren't
            // actually the same pose — Hunyuan3D-2mv's multiview
            // conditioning assumes all three ARE one pose from three camera
            // angles, and feeding it three different poses/styles produces
            // exactly this kind of geometry conflict). This final check
            // shows all three together and asks whether they're actually
            // usable as one consistent set — pose, art style, and
            // appearance all need to roughly agree, not just the subject.
            if (frontUsed && leftUsed && backUsed && await verifyImagesAreConsistentSet(frontPath, leftPath, backPath)) {
              multiviewPaths = { front: frontPath, left: leftPath, back: backPath };
              referenceLabel = `three separately-searched angle photos of "${subject}" (front: ${frontUsed}, side: ${leftUsed}, back: ${backUsed})`;
            }
          }
          } // !wantsSingleImageOnly

          if (!multiviewPaths) {
            // Image-to-3D generation is highly sensitive to input quality —
            // a clean, isolated-subject photo produces far better geometry
            // than a random cluttered/watermarked web photo. Bias the
            // search toward that first, falling back to a plain search if
            // it finds nothing.
            let imageUrls = await webImageSearch(`${subject} isolated on white background`);
            if (imageUrls.length === 0) imageUrls = await webImageSearch(subject);
            if (imageUrls.length === 0) {
              imageReplyText = `I couldn't find a usable reference image for "${subject}" to build a model from.`;
            } else {
              const downloadPath = path.join(IMAGE23D_TMP_DIR, `${Date.now()}.jpg`);
              const usedUrl = await downloadFirstAvailableImage(imageUrls, downloadPath);
              if (usedUrl) {
                imagePath = downloadPath;
                referenceLabel = `this reference image: ${usedUrl}`;
                tempFiles.push(imagePath);
              } else {
                imageReplyText = `Found a reference image for "${subject}" but couldn't download it to work with.`;
              }
            }
          }
        }

        if (imagePath || multiviewPaths) {
          // Needs exclusive use of the GPU on an 8GB card — see
          // withGpuExclusive's comment for why and what that costs.
          const meshResult = await withGpuExclusive(() => multiviewPaths
            ? generateMeshWithHunyuan3DMultiview(multiviewPaths!.front, multiviewPaths!.left, multiviewPaths!.back)
            : generateMeshWithHunyuan3D(imagePath!));
          if (!meshResult.success) {
            imageReplyText = `Generating a mesh for "${subject}" failed: ${meshResult.error}`;
          } else {
            tempFiles.push(meshResult.meshPath!);
            let meshPathForImport = meshResult.meshPath!;
            let scaleNote = "Heads up: this mesh isn't real-world scaled, so it may look tiny or huge at first — Fusion's Scale tool can fix that, and geometry detail is still rough compared to a hand-modeled asset.";
            if (targetSizeCm !== null) {
              const scaledPath = await scaleMeshToTargetSize(meshResult.meshPath!, targetSizeCm);
              if (scaledPath) {
                meshPathForImport = scaledPath;
                tempFiles.push(scaledPath);
                scaleNote = `Scaled it so its largest dimension is about ${targetSizeCm}cm — since I can't know which axis the model treats as "height", that's an overall-size match rather than a guaranteed exact height.`;
              } else {
                scaleNote = `Tried to scale it to ${targetSizeCm}cm but the scaling step failed, so it's still at the model's raw output size — Fusion's Scale tool can fix that manually.`;
              }
            }
            let repairNote = "";
            const repairResult = await repairMesh(meshPathForImport);
            if (repairResult === null) {
              repairNote = " (Couldn't run the mesh validation/repair check, so this is unvalidated — worth a closer look in Fusion before printing.)";
            } else if (!repairResult.report.usable) {
              imageReplyText = `Generated a mesh for "${subject}", but it came out too broken to use: ${repairResult.report.reason}. Try a different reference image, or a more distinct/isolated subject.`;
            } else {
              meshPathForImport = repairResult.outputPath;
              tempFiles.push(repairResult.outputPath);
              if (repairResult.report.componentDroppedCount > 0) {
                repairNote = ` Dropped ${repairResult.report.componentDroppedCount} disconnected stray piece(s) that didn't belong to the main shape.`;
              } else if (repairResult.report.repaired) {
                repairNote = " Repaired some holes in the mesh before import.";
              }
            }

            // repairResult.report.usable === false is a hard stop — a
            // known-broken mesh shouldn't reach Fusion at all. A null
            // repairResult (the check itself failed to run) is a soft
            // failure, so that case still falls through to import.
            if (!(repairResult !== null && !repairResult.report.usable)) {
              const importResult = await importMeshToFusion(meshPathForImport);
              imageReplyText = importResult.success
                ? `Done — imported a 3D model of "${subject}" into Fusion 360, built from ${referenceLabel}.\n\n${scaleNote}${repairNote}`
                : `Fusion rejected importing the mesh: ${importResult.error}`;

              // Best-effort: the preview panel is a nice-to-have, not worth
              // failing an otherwise-successful import over. Converts from
              // meshPathForImport (post-scaling and post-repair, if any) so
              // the preview matches what actually landed in Fusion, not the
              // raw pre-scale/pre-repair output.
              if (importResult.success) {
                const glbFilename = `${Date.now()}.glb`;
                const glbPath = path.join(MODELS_DIR, glbFilename);
                if (await convertMeshToGlb(meshPathForImport, glbPath)) {
                  generatedModel = registerGeneratedModel(subject, glbFilename);
                }

                // Best-effort, same reasoning as the GLB preview above — a
                // failed/unconfigured slicer shouldn't sink an otherwise
                // successful generation. Deliberately NOT pushed to
                // tempFiles: the .gcode.3mf needs to survive past this
                // request until the user confirms or cancels the print.
                const sliceResult = await sliceModel(meshPathForImport);
                if (sliceResult.success && sliceResult.gcodePath) {
                  pendingPrintJob = {
                    gcodePath: sliceResult.gcodePath,
                    subject,
                    estimatedTimeMin: sliceResult.estimatedTimeMin ?? null,
                    estimatedFilamentG: sliceResult.estimatedFilamentG ?? null,
                    createdAt: Date.now(),
                  };
                  const estimateText = sliceResult.estimatedTimeMin
                    ? ` Estimated print time: ${sliceResult.estimatedTimeMin} min${sliceResult.estimatedFilamentG ? `, ~${sliceResult.estimatedFilamentG}g filament` : ""}.`
                    : "";
                  imageReplyText += `\n\nSliced and ready to print.${estimateText} Reply "print" to hand it off to Bambu Connect, or ignore this to skip.`;
                } else if (sliceResult.error) {
                  imageReplyText += `\n\n(Skipped printing: ${sliceResult.error})`;
                }
              }
            }
          }
        }

        cleanupTempFiles(tempFiles);
      }

      conversationHistory.push(`Assistant: ${imageReplyText}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({
        response: imageReplyText,
        hasProposedChanges: false,
        ...(generatedModel ? {
          modelId: generatedModel.id,
          modelUrl: `/generated-models/${generatedModel.filename}`,
          modelSubject: generatedModel.subject
        } : {})
      });
    }

    // Edits an already-generated model in place — scale/mirror/simplify/
    // fill-holes on whichever model is open in the viewer panel (or the
    // most recent one if none is open — see resolveEditTargetModel).
    // Preview-only: doesn't re-run Fusion import or slicing, same posture
    // as scaleMeshToTargetSize/repairMesh being pure mesh-to-mesh steps
    // with no side effects of their own.
    if (looksLikeMeshEditRequest) {
      const target = resolveEditTargetModel(openModelId);
      let editReplyText: string;
      let updatedModel: GeneratedModelEntry | null = null;

      if (!target) {
        editReplyText = "There's no generated model to edit yet — generate one first, or open one from the viewer.";
      } else {
        const sourcePath = path.join(MODELS_DIR, target.filename);
        fs.mkdirSync(IMAGE23D_TMP_DIR, { recursive: true });
        const tmpOut = path.join(IMAGE23D_TMP_DIR, `${Date.now()}_edited.glb`);

        let ok = false;
        let note = "";
        if (MESH_FILL_HOLES_PATTERN.test(message)) {
          const repairResult = await repairMesh(sourcePath);
          ok = !!repairResult?.report.usable;
          if (ok) fs.copyFileSync(repairResult!.outputPath, tmpOut);
        } else if (MESH_MIRROR_PATTERN.test(message)) {
          ok = await editMesh(sourcePath, tmpOut, "mirror", extractMirrorAxis(message));
        } else if (MESH_SIMPLIFY_PATTERN.test(message)) {
          ok = await editMesh(sourcePath, tmpOut, "simplify-percent", String(extractSimplifyPercent(message)));
          // Confirmed directly: simplify_quadric_decimation strips UV/
          // material data — a textured mesh comes out visibly blank/noisy,
          // not just lower-detail. Warn rather than silently degrade.
          if (ok) note = " Note: simplifying removes color/texture — the model will look bare afterward.";
        } else if (MESH_SCALE_PATTERN.test(message)) {
          const sizeCm = extractTargetSizeCm(message);
          if (sizeCm !== null) {
            const scaled = await scaleMeshToTargetSize(sourcePath, sizeCm);
            ok = !!scaled;
            if (ok) fs.copyFileSync(scaled!, tmpOut);
          } else {
            const factor = extractScaleFactor(message);
            ok = factor !== null && await editMesh(sourcePath, tmpOut, "scale-factor", String(factor));
          }
        }

        if (ok) {
          updatedModel = replaceGeneratedModelFile(target.id, tmpOut);
          editReplyText = updatedModel
            ? `Done — updated "${updatedModel.subject}".${note} This is a preview-only change — re-import or re-slice separately if you want to print it.`
            : "The edit succeeded but the registry update failed — the model may be out of sync.";
        } else {
          editReplyText = "Couldn't apply that edit — the original model is unchanged.";
        }
      }

      conversationHistory.push(`Assistant: ${editReplyText}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({
        response: editReplyText,
        hasProposedChanges: false,
        ...(updatedModel ? {
          modelId: updatedModel.id,
          modelUrl: `/generated-models/${updatedModel.filename}`,
          modelSubject: updatedModel.subject
        } : {})
      });
    }

    // An attached image with no 3D-generation intent is a general "what is
    // this" / "describe this" question — send it straight to the local
    // vision-capable model rather than the JSON-modification/recall pipeline
    // below, which has nothing to do with image content.
    if (hasAttachedImage) {
      // Explicit and blunt on purpose: this model has a strong trained habit
      // of reflexively claiming "I'm text-only, I can't see images" even
      // when it demonstrably did receive and correctly process one (observed
      // directly — it described real image content accurately, then denied
      // being able to see anything). Left implicit, that denial reflex wins
      // even over correct visual analysis in the same response.
      const visionPrompt = `${personality}\n\nAn image is attached to this message and you can see it directly — for this request you have real vision input, not just text. Describe or answer based on what is actually visible. Never claim you can't see images or that you're text-only; that would be false here.\n\nUser: ${message}`;
      let visionReply: string;
      try {
        visionReply = await callOllama(visionPrompt, 500, false, [attachedImage!.base64]);
      } catch (err: any) {
        visionReply = `I had trouble looking at that image: ${err.message}`;
      }

      conversationHistory.push(`Assistant: ${visionReply}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({ response: visionReply, hasProposedChanges: false });
    }

    const recentHistory = conversationHistory.slice(-20).join("\n");

    let recallContext = "";
    if (isRecallQuery) {
      const channelLabel = recallChannel ? " (scoped to Telegram messages only)" : "";
      if (recallDate) {
        const sessions = findArchivesByDate(recallDate)
          .map(s => ({
            ...s,
            messages: stripStaleNoMemoryReplies(s.messages).filter(m => !recallChannel || m.includes("(Telegram)"))
          }))
          .filter(s => s.messages.length > 0);
        const dateLabel = recallDate.toDateString();
        recallContext = sessions.length > 0
          ? `--- ARCHIVED CONVERSATION FROM ${dateLabel}${channelLabel} ---\n` +
            sessions.map(s => s.messages.join("\n")).join("\n---\n") +
            `\n--- END ARCHIVED CONVERSATION ---\n` +
            `Every message between the markers above happened on ${dateLabel} — that is the true date of this whole block, no matter what other dates get mentioned in the conversation text itself (the user may have been asking about a different day within it). ` +
            `Answer the user's question about ${dateLabel} by summarizing what these messages show. Do not claim you have no record of ${dateLabel}: you are looking at it right now.\n\n`
          : `--- NOTE: No archived conversation${channelLabel} was found for ${dateLabel}. Tell the user you have no record of that day${channelLabel}. ---\n\n`;
      } else {
        // Strip the channel phrase itself before extracting the topic
        // keyword — otherwise "what did we talk about pineapple on
        // telegram" would extract "pineapple on telegram" as the literal
        // search keyword (extractTopicKeyword's "about ... to end of
        // string" regex has no way to know "on telegram" is a qualifier,
        // not part of the topic), which then matches nothing.
        const topicSource = recallChannel ? message.replace(CHANNEL_TRIGGER_PATTERN, "").trim() : message;
        const topic = extractTopicKeyword(topicSource);
        const hits = (topic.length >= 3 ? searchArchivesByKeyword(topic) : [])
          .filter(h => !STALE_NO_MEMORY_REPLY_PATTERN.test(h.line))
          .filter(h => !recallChannel || h.line.includes("(Telegram)"))
          .slice(0, 30);
        recallContext = hits.length > 0
          ? `--- ARCHIVED MENTIONS OF "${topic}"${channelLabel} ---\n` +
            hits.map(h => `[${new Date(h.date).toDateString()}] ${h.line}`).join("\n") +
            `\n--- END ARCHIVED MENTIONS ---\nAnswer the user's question using the archived mentions above.\n\n`
          : `--- NOTE: No archived mentions of "${topic}"${channelLabel} were found. Tell the user you have no record of discussing that${channelLabel}. ---\n\n`;
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
    } else if (looksLikeStatusQuery) {
      console.log("FETCHING LIVE SYSTEM STATUS");
      const [ollamaHealthy, searxngHealthy, speechHealthy] = await Promise.all([
        isServiceHealthy(OLLAMA_URL.replace(/\/api\/generate$/, "/api/version")),
        isServiceHealthy(`${SEARXNG_URL}/healthz`),
        isServiceHealthy(`${SPEECH_SERVICE_URL}/health`)
      ]);
      const statusServices = [
        { name: "Ollama (language model)", healthy: ollamaHealthy },
        { name: "SearXNG (web search)", healthy: searxngHealthy },
        { name: "Speech service (voice input/output)", healthy: speechHealthy }
      ];
      const healthyCount = statusServices.filter(s => s.healthy).length;
      const coreIntegrityPercent = (healthyCount / statusServices.length) * 100;
      const gpu = getGpuStats();

      recallContext = `--- LIVE SYSTEM STATUS ---\n` +
        `Core Integrity: ${coreIntegrityPercent.toFixed(2)}% (${healthyCount}/${statusServices.length} services healthy)\n` +
        statusServices.map(s => `- ${s.name}: ${s.healthy ? "UP" : "DOWN"}`).join("\n") +
        (gpu
          ? `\nNeural Load (GPU utilization): ${gpu.utilizationPercent}%\nModel Temp (GPU temperature): ${gpu.temperatureC}°C\nVRAM: ${gpu.vramUsedMB}MB / ${gpu.vramTotalMB}MB`
          : `\nGPU stats unavailable (no NVIDIA GPU detected, or nvidia-smi isn't on PATH).`) +
        `\n--- END LIVE SYSTEM STATUS ---\n` +
        `Answer the user's question using this real, current status data. If any service is DOWN, name it specifically. Do not guess or make up different numbers.\n\n`;

      console.log("LIVE STATUS CONTEXT:");
      console.log(recallContext);
    } else if (looksLikeVisionStatusQuery) {
      console.log("FETCHING LIVE VISION STATUS");
      let visionStatusFact: string;
      try {
        const visionRes = await fetch(`${VISION_SERVICE_URL}/status`, { signal: AbortSignal.timeout(1500) });
        if (!visionRes.ok) throw new Error(`vision service returned ${visionRes.status}`);
        const { identity, camera_ok } = await visionRes.json() as { identity: string | null; camera_ok: boolean };
        visionStatusFact = !camera_ok
          ? "The camera itself can't currently be accessed (it may be in use by another application, like the Windows Camera app, or physically disconnected) — there's no live view to report on."
          : identity === null
            ? "The camera does not currently see anyone in view."
            : identity === "unknown"
              ? "The camera currently sees someone in view, but their face isn't one that's been enrolled/recognized."
              : `The camera currently sees ${identity} in view.`;
      } catch {
        visionStatusFact = "The vision service isn't currently running or reachable, so there's no live camera data available right now.";
      }

      recallContext = `--- LIVE VISION STATUS ---\n${visionStatusFact}\n--- END LIVE VISION STATUS ---\n` +
        `Answer the user's question using this real, current camera data, and nothing else — do not blend it with guesses or unrelated conversation history. These are three distinct, mutually exclusive situations; say specifically which one applies, in your own words: (1) the camera itself is inaccessible right now (e.g. blocked by another app) — this is NOT the same as nobody being there, say explicitly that the camera can't be reached; (2) the camera works and currently sees nobody; (3) the camera works and sees someone (named, or unrecognized). Do not invent a name.\n\n`;

      console.log("LIVE VISION STATUS CONTEXT:");
      console.log(recallContext);
    } else if (looksLikeHardwareQuery) {
      console.log("FETCHING LIVE HARDWARE STATS");
      const gpu = getGpuStats();
      const hardwareFact = gpu
        ? `GPU: NVIDIA card detected. VRAM: ${gpu.vramUsedMB}MB used / ${gpu.vramTotalMB}MB total (${(gpu.vramTotalMB / 1024).toFixed(1)}GB total). Utilization: ${gpu.utilizationPercent}%. Temperature: ${gpu.temperatureC}°C.`
        : "No NVIDIA GPU was detected (or nvidia-smi isn't on PATH) — this only supports NVIDIA cards right now, the same limitation the HUD's own GPU stats have.";

      recallContext = `--- LIVE HARDWARE STATUS ---\n${hardwareFact}\n--- END LIVE HARDWARE STATUS ---\n` +
        `Answer the user's question using this real, current hardware data. Do not guess or make up different numbers — if no GPU was detected, say so plainly rather than inventing specs.\n\n`;

      console.log("LIVE HARDWARE CONTEXT:");
      console.log(recallContext);
    } else if (looksLikeModelRecommendQuery) {
      console.log("FETCHING MODEL RECOMMENDATION DATA");
      const gpu = getGpuStats();
      const pulled = await listPulledModels();
      const recommendation = buildModelRecommendationData(pulled, gpu);

      // Each pulled model is labeled fits/exceeds directly in the data
      // (by buildModelRecommendationData) instead of leaving that arithmetic
      // to the LLM — confirmed live that leaving it implicit let the model
      // recommend a 23.9GB model as its "top pick" on a 15.9GB card, backed
      // by a fabricated "it's probably quantized or offloaded" excuse with
      // no real data behind it. The measured size already reflects whatever
      // quantization that model was pulled in — there's nothing more to
      // offload it into.
      const pulledSummary = recommendation.pulled.length > 0
        ? recommendation.pulled.map(m => {
            const fitNote = m.fits === null
              ? "VRAM fit unknown, no GPU detected"
              : m.fits
                ? "fits your VRAM"
                : "EXCEEDS your VRAM — do not recommend this as a primary pick, it will run slow or fail to load fully on the GPU";
            return `- ${m.name} (already pulled, ${m.sizeGB.toFixed(1)}GB on disk — this is real, measured — ${fitNote})`;
          }).join("\n")
        : "(no models currently pulled)";

      const suggestionSummary = recommendation.suggestions.length > 0
        ? recommendation.suggestions.map(m => `- ${m.name} (approx. ${m.approxVramGB}GB — NOT pulled, NOT measured, a rough estimate only)`).join("\n")
        : "(no additional suggestions — either no GPU was detected, or nothing in the reference list clearly fits)";

      const hardwareLine = recommendation.totalVramGB !== null
        ? `Total VRAM detected: ${recommendation.totalVramGB.toFixed(1)}GB.`
        : "No NVIDIA GPU was detected — recommend conservative, CPU-friendly models (small parameter counts) rather than reasoning about VRAM fit at all.";

      recallContext = `--- MODEL RECOMMENDATION DATA ---\n${hardwareLine}\n\nAlready pulled locally (real, measured sizes):\n${pulledSummary}\n\nOther models that would likely fit, if you wanted to pull one (APPROXIMATE — not measured, a rough guide only, manually curated and may be out of date for newer model releases):\n${suggestionSummary}\n--- END MODEL RECOMMENDATION DATA ---\n` +
        `Answer the user's question using this real data. Clearly distinguish already-pulled models (real sizes) from suggestions (approximate, explicitly tell the user these are estimates, not guarantees) — do not present a suggestion as if it were measured. Your top pick MUST be one that fits VRAM (marked "fits your VRAM" above) — never recommend a model marked as exceeding VRAM as a primary/top pick, and never speculate that an oversized model "probably still works" via quantization or offloading; the size shown already accounts for that, so say plainly that it exceeds VRAM and would run slow or fail instead.\n\n`;

      console.log("MODEL RECOMMENDATION CONTEXT:");
      console.log(recallContext);
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
    } else if (looksLikeDocumentQuestion) {
      const activeDocument = documentStore.get(currentDocumentId!)!;
      const relevantChunks = await retrieveRelevantChunks(currentDocumentId!, message, 5);

      console.log("RETRIEVING FROM DOCUMENT:", activeDocument.fileName, "| chunks found:", relevantChunks.length);

      recallContext = relevantChunks.length > 0
        ? `--- RELEVANT EXCERPTS FROM "${activeDocument.fileName}" ---\n` +
          relevantChunks.map((c, i) => `[Excerpt ${i + 1}]\n${c}`).join("\n\n") +
          `\n--- END EXCERPTS ---\n` +
          // Explicit and blunt on purpose, same reasoning as the vision
          // prompt above: this model has a trained habit of reflexively
          // denying capabilities it actually has in this exact turn.
          `Answer the user's question using these excerpts from the attached document — you have direct access to them, this is not secondhand information. Never claim you can't read documents or don't have access to it; that would be false here. If the excerpts don't actually answer the question, say so honestly instead of guessing.\n\n`
        : `--- NOTE: No relevant excerpts were found in "${activeDocument.fileName}" for this question. ---\n\n`;
    }

    // A "live status" answer is defined entirely by current state, not by
    // consistency with what was said before — but recentHistory gets baked
    // into every other prompt unconditionally, and confirmed directly: after
    // several turns of the camera genuinely being blocked, the model kept
    // repeating "camera inaccessible" even once given a fresh fact saying it
    // now works, anchoring on its own prior answers over the live data. For
    // these two intents specifically, omit history so the fresh fact is the
    // only thing in the room.
    const isFreshLiveStatusQuery = looksLikeStatusQuery || looksLikeVisionStatusQuery || looksLikeHardwareQuery || looksLikeModelRecommendQuery;
    const fullPrompt = wantsModification
      ? `System Instruction:\n${metaSystemInstruction}\n\n` + `User Request:\n${message}`
      : isFreshLiveStatusQuery
        ? `System Instruction:\n${metaSystemInstruction}\n\n` + recallContext + `User Request:\n${message}`
        : `System Instruction:\n${metaSystemInstruction}\n\n` + recallContext + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;
    
    console.log("WANTS MODIFICATION:", wantsModification);
    console.log("PROMPT SENT TO MODEL:");
    console.log(fullPrompt);

    console.log(`[MODEL] ${selectedModel} | complex=${isComplex}`);

   
    let aiResponse = await callOllama(fullPrompt, isComplex ? 3000 : 300, wantsDeeperThinking);

    console.log("RAW AI RESPONSE:");
    console.log(aiResponse);


    const UNCERTAINTY_PATTERN = /\b(i don'?t know|i do not know|i'?m not (sure|aware|certain)|i am not (sure|aware|certain)|i (can'?t|cannot) (verify|confirm|access)|as of my (last|knowledge)|i (don'?t|do not) have (\w+\s+){0,3}(access|information|real-time|up-to-date|current|internet|any (record|knowledge)|the (latest|current))|beyond my (knowledge|training)|no (real-time|live) (access|data)|i'?m unable to (access|verify|confirm)|i am unable to (access|verify|confirm))\b/i;

    const searchMatch = aiResponse.match(SEARCH_TAG_PATTERN);
    const soundsUncertain = isPlainChatTurn && !searchMatch && UNCERTAINTY_PATTERN.test(aiResponse);

    if (searchMatch || soundsUncertain) {
      let searchQuery: string;
      if (searchMatch) {
        searchQuery = searchMatch[1].trim();
        console.log("WEB SEARCH REQUESTED (via tag):", searchQuery);
      } else {
        console.log("MODEL SOUNDED UNCERTAIN — retrying with search. Raw response was:", aiResponse);
        const queryExtractionPrompt = `Extract a short, focused web search query (3-8 words, no quotes or extra punctuation) that would find the answer to this request. Reply with ONLY the query text and nothing else.\n\nRequest: "${message}"`;
        const rawQuery = await callOllama(queryExtractionPrompt, 30);
        searchQuery = rawQuery.replace(/["'\n]+/g, " ").trim() || message;
        console.log("EXTRACTED QUERY:", searchQuery);
      }

      const results = await webSearch(searchQuery);
      const searchContext = results.length > 0
        ? `--- WEB SEARCH RESULTS FOR "${searchQuery}" ---\n` +
          results.map((r, i) => `${i + 1}. ${r.title}\n${r.description}\n${r.url}`).join("\n\n") +
          `\n--- END WEB SEARCH RESULTS ---\n` +
          `Answer the user's original question using these results. If they don't actually answer it, say so honestly instead of guessing. ` +
          `Don't list the URLs unless asked — but if the user later asks for the link(s)/source(s), use the exact URLs shown above. Never invent a URL that isn't listed there.\n\n`
        : `--- NOTE: The web search for "${searchQuery}" returned no results (or web search isn't configured). Tell the user you couldn't find current information on this. ---\n\n`;

      // Keep these results (with real URLs) available for a follow-up like "give me the link to the sites you used".
      stickySearchContext = searchContext;
      stickySearchTurnsRemaining = STICKY_RECALL_FOLLOWUP_TURNS;

      const followUpPrompt = `System Instruction:\n${metaSystemInstruction}\n\n` + searchContext + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;

      aiResponse = await callOllama(followUpPrompt, isComplex ? 3000 : 400, wantsDeeperThinking);

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

    // Noah is sometimes asked to modify memory/personality but just chats back without actually proposing any changes
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

        conversationHistory.push(`Assistant${channelTag}: ${historySafeResponse}`);
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
      diff: gitDiff,
      // Lets the frontend clear a stale chip and prompt re-attachment
      // instead of silently resending a dead id forever (e.g. after a
      // server restart, since documentStore is in-memory only).
      ...(currentDocumentId && !hasActiveDocument ? { documentNotFound: true } : {})
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
  stopVisionService();
  process.exit(0);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
// Node documents that Windows delivers SIGHUP when the console window itself
// is closed (as opposed to Ctrl+C, which is SIGINT) — without this, closing
// the terminal window leaves the speech/vision child processes running as
// orphans (camera and mic still active) even though the main server is gone.
process.on("SIGHUP", () => shutdown("SIGHUP"));
// Last-resort backstop for any other exit path (an uncaught exception, or
// something elsewhere calling process.exit() directly without going through
// shutdown() above) — kill() is fire-and-forget, so it's safe to call here
// even though 'exit' handlers can't do async work.
process.on("exit", () => {
  stopSpeechService();
  stopVisionService();
});