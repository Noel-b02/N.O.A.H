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
const FUSION_BRIDGE_URL = process.env.FUSION_BRIDGE_URL ?? "http://localhost:9000";

const IMAGE23D_DIR = path.join(__dirname, "image23d");
const IMAGE23D_TMP_DIR = path.join(IMAGE23D_DIR, "tmp");
const TRIPOSR_DIR = path.join(IMAGE23D_DIR, "TripoSR");
const TRIPOSR_PYTHON = path.join(TRIPOSR_DIR, "venv", "Scripts", "python.exe");

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

// Pulls the named subject out of an image-based Fusion request (e.g. "create
// a 3d model of spiderman in fusion" -> "spiderman") so it can be used as
// both the image search query and the reply text, without needing a
// dedicated Ollama call just to extract a noun phrase.
// Matches a trailing sizing instruction (e.g. ", make it 5cm", "5cm tall",
// "about 2 inches") so it can be stripped before subject extraction — left
// in place, it either gets glued onto the subject text (breaking the image
// search query) or its own "make" verb gets caught by the verb-stripping
// regex below, leaving a mangled leftover like "it 5cm".
const SIZE_CLAUSE_PATTERN = /,?\s*\b(?:make (?:it|the model)\s+|about\s+|roughly\s+|sized?\s+(?:at\s+)?|at\s+)?\d+(?:\.\d+)?\s?(?:millimeters?|mm|centimeters?|cm|meters?|m|inches?|in|feet|ft)\b(?:\s+(?:tall|high|wide|long|in\s+(?:height|size)))?/i;

function extractFusionSubject(message: string): string {
  const withoutSize = message.replace(SIZE_CLAUSE_PATTERN, "").trim();
  const ofMatch = withoutSize.match(/\bof\s+(?:a|an|the)\s+(.+?)[\?\.!]*$/i) ?? withoutSize.match(/\bof\s+(.+?)[\?\.!]*$/i);
  const raw = ofMatch ? ofMatch[1] : withoutSize.replace(/\b(create|make|generate|build|design)\b/gi, "").replace(/\b(a|an|the)\b/gi, "");
  // The "of ..." match is greedy to end-of-string, so it swallows trailing
  // trigger phrases too (e.g. "of a fox in fusion 360" -> "fox in fusion
  // 360") — strip those out regardless of which branch produced the string.
  return raw
    .replace(/\b(3d models?|3d shapes?|cad models?|in fusion(?: ?360)?)\b/gi, "")
    .trim();
}

// Looks for a "<number> <unit>" size mention (e.g. "5cm", "2 inches", "10mm
// tall") and converts it to centimeters — the unit the Fusion import step
// already treats TripoSR's raw output numbers as. Requires the unit to
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

// Rescales the generated mesh (via trimesh, already installed for TripoSR)
// so its largest bounding-box dimension matches the requested size, before
// it ever reaches Fusion — sidesteps relying on Fusion's own scale tooling
// supporting mesh bodies, which isn't clearly documented either way. Writes
// to a separate output file rather than overwriting the input: if this step
// fails partway, the original mesh must still be importable as a fallback.
function scaleMeshToTargetSize(meshPath: string, targetSizeCm: number): Promise<string | null> {
  return new Promise((resolve) => {
    const scaledPath = meshPath.replace(/\.obj$/i, "_scaled.obj");
    const child = spawn(TRIPOSR_PYTHON, [path.join(IMAGE23D_DIR, "scale_mesh.py"), meshPath, String(targetSizeCm), scaledPath]);
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
      .filter((src: unknown): src is string => typeof src === "string" && /^https?:\/\//.test(src));
  } catch (err) {
    console.error("Web image search error (is SearXNG running at " + SEARXNG_URL + "?):", err);
    return [];
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
    const buffer = Buffer.from(await res.arrayBuffer());
    fs.mkdirSync(path.dirname(destPath), { recursive: true });
    fs.writeFileSync(destPath, buffer);
    return true;
  } catch (err) {
    console.error("Reference image download error:", imageUrl, err);
    return false;
  }
}

interface MeshGenerationResult {
  success: boolean;
  meshPath?: string;
  error?: string;
}

// Runs TripoSR (image23d/TripoSR) as a one-off child process rather than a
// persistent service — this is a rarely-used, GPU-heavy step, and keeping it
// out-of-process means it doesn't hold VRAM hostage from Ollama/Kokoro the
// rest of the time. See IMAGE23D_SETUP.md for the venv this expects.
function generateMeshFromImage(imagePath: string): Promise<MeshGenerationResult> {
  return new Promise((resolve) => {
    if (!fs.existsSync(TRIPOSR_PYTHON)) {
      resolve({ success: false, error: `TripoSR venv not found at ${TRIPOSR_PYTHON} — see IMAGE23D_SETUP.md.` });
      return;
    }

    const outputDir = path.join(TRIPOSR_DIR, "output");
    const child = spawn(TRIPOSR_PYTHON, ["run.py", imagePath, "--output-dir", outputDir], { cwd: TRIPOSR_DIR });

    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d.toString(); });

    const timeout = setTimeout(() => {
      child.kill();
      resolve({ success: false, error: "TripoSR generation timed out after 3 minutes." });
    }, 180000);

    child.on("error", (err) => {
      clearTimeout(timeout);
      resolve({ success: false, error: `Failed to launch TripoSR: ${err.message}` });
    });

    child.on("exit", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        resolve({ success: false, error: stderr.trim().slice(-2000) || `TripoSR exited with code ${code}` });
        return;
      }
      const meshPath = path.join(outputDir, "0", "mesh.obj");
      if (!fs.existsSync(meshPath)) {
        resolve({ success: false, error: "TripoSR finished but no mesh.obj was produced." });
        return;
      }
      resolve({ success: true, meshPath });
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
// fake fluctuations — CORE INTEGRITY reflects whether the backing services
// (Ollama, SearXNG, the speech service) are actually reachable, and NEURAL
// LOAD / MODEL TEMP read genuine GPU utilization and temperature.
app.get('/api/hud-metrics', async (_, res) => {
  const [ollamaHealthy, searxngHealthy, speechHealthy] = await Promise.all([
    isServiceHealthy(OLLAMA_URL.replace(/\/api\/generate$/, "/api/version")),
    isServiceHealthy(`${SEARXNG_URL}/healthz`),
    isServiceHealthy(`${SPEECH_SERVICE_URL}/health`)
  ]);

  const services = { ollama: ollamaHealthy, searxng: searxngHealthy, speech: speechHealthy };
  const healthyCount = Object.values(services).filter(Boolean).length;
  const coreIntegrityPercent = (healthyCount / Object.keys(services).length) * 100;

  res.json({
    coreIntegrityPercent,
    services,
    gpu: getGpuStats()
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

    if (!speechServiceProcess) {
      console.warn("[speech] Speech service process is down — attempting to restart it.");
      startSpeechService().catch(e => console.error("[speech] Restart attempt failed:", e));
    }

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

  // "No changes to your personality" / "I didn't change anything" mention the
  // keyword while explicitly saying nothing should happen — without this,
  // they trip wantsModification just as wrongly as "do you remember" tripped
  // it on the word "remember" (see REMEMBER_QUESTION_PATTERN above).
  const NEGATED_MODIFICATION_PATTERN = /\b(no|not|don'?t|didn'?t|without|never)\s+(\w+\s+){0,2}(changes?|modif(y|ication)|updates?)\b/i;

  const wantsModification = !NEGATED_MODIFICATION_PATTERN.test(message) && (
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

  // A follow-up on an already-answered recall query (e.g. "give me specifics", "quote it") — doesn't re-trigger isRecallQuery on its own, but should still get the archived context and the smarter model.
  const isStickyRecallFollowup = !wantsModification && !isRecallQuery && stickyRecallTurnsRemaining > 0;

  // "What does core integrity mean" is answered fine from the general explanation now in personality.txt — but "why isn't it 100%" or "what's down" needs the actual live health state, which the model has no way to
  // know on its own. This always fetches fresh (no sticky reuse): unlike a search result, service health can flip within seconds, so reusing stale
  // status from a previous turn would risk telling the user a service is down when it's since recovered, or vice versa.
  const STATUS_TRIGGER_PATTERN = /\b(core integrity|neural load|model temp|diverg(ence)?[\s_-]?buf(fer)?|system status|what'?s down|which service|is (everything|anything) (down|broken|working|up)|are you (down|working|online)|health check)\b/i;
  const looksLikeStatusQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && STATUS_TRIGGER_PATTERN.test(message);

  // "Give me today's headlines" is a news-digest request, not a factual
  // lookup — it gets its own trigger and its own fetch path (SearXNG's news
  // category + day filter) rather than a plain web search, which just
  // surfaces static homepages for a bare term like "headlines".
  const HEADLINES_TRIGGER_PATTERN = /\b(headlines|breaking news|today'?s news|news today|on the news|what'?s (been )?reported|happening in the world)\b/i;
  const looksLikeHeadlinesQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && HEADLINES_TRIGGER_PATTERN.test(message);

  const SEARCH_TRIGGER_PATTERN = /\b(latest news|who won|release date|launch date|premiere date|when('s| is| does| will).{0,30}(come out|coming out|releas(e|ing)|drop(ping)?|launch(ing)?|premier(e|ing))|current (weather|price|score|exchange rate)|how much (is|does|would)|price of|what'?s the weather|weather (today|forecast|right now))\b/i;
  const looksLikeSearchQuery = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeHeadlinesQuery && SEARCH_TRIGGER_PATTERN.test(message);

  // A follow-up on an already-answered search or headlines request (e.g.
  // "give me the link to the sites you used", "which one said that") —
  // needs the same results (with real URLs) rather than the model
  // improvising from its own summary.
  const isStickySearchFollowup = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && stickySearchTurnsRemaining > 0;

  // Requires an explicit "3d model"/"cad model"/"in fusion" mention alongside
  // a creation verb — kept deliberately narrow since this triggers real code
  // execution inside a running Fusion 360 session, not just a chat response.
  const FUSION_TRIGGER_PATTERN = /\b(create|make|generate|build|design|remove|delete|clear)\b.{0,40}\b(3d models?|3d shapes?|cad models?|in fusion(?: ?360)?)\b/i;
  const looksLikeFusionRequest = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !isStickySearchFollowup && FUSION_TRIGGER_PATTERN.test(message);

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

  const isComplex = wantsModification || isRecallQuery || isStickyRecallFollowup || looksLikeStatusQuery || looksLikeHeadlinesQuery || looksLikeSearchQuery || isStickySearchFollowup || looksLikeFusionRequest || (
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
  const isPlainChatTurn = !wantsModification && !isRecallQuery && !isStickyRecallFollowup && !looksLikeStatusQuery && !looksLikeHeadlinesQuery && !looksLikeSearchQuery && !isStickySearchFollowup && !looksLikeFusionRequest;

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

    async function callOllama(prompt: string, numPredict: number, think: boolean = false): Promise<string> {
      const res = await fetch(OLLAMA_URL, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          prompt,
          stream: false,
          think,
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
        const bridgeAvailable = await isServiceHealthy(`${FUSION_BRIDGE_URL}/health`);

        if (!bridgeAvailable) {
          fusionReplyText = `I generated code for this, but couldn't reach the Fusion 360 bridge at ${FUSION_BRIDGE_URL} — make sure Fusion 360 is running with the NoahFusionBridge add-in active.\n\nGenerated code:\n${generatedCode}`;
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
    // reference photo, run it through the local TripoSR model to get a
    // mesh, then import that mesh directly into the Fusion document.
    if (looksLikeImageFusionRequest) {
      const subject = extractFusionSubject(message);
      const targetSizeCm = extractTargetSizeCm(message);
      let imageReplyText: string;

      const bridgeAvailable = await isServiceHealthy(`${FUSION_BRIDGE_URL}/health`);
      if (!bridgeAvailable) {
        imageReplyText = `I'd need the Fusion 360 bridge running to import a model of "${subject}" — make sure Fusion 360 is running with the NoahFusionBridge add-in active.`;
      } else {
        const imageUrls = await webImageSearch(subject);
        if (imageUrls.length === 0) {
          imageReplyText = `I couldn't find a usable reference image for "${subject}" to build a model from.`;
        } else {
          const imagePath = path.join(IMAGE23D_TMP_DIR, `${Date.now()}.jpg`);
          const downloaded = await downloadImageToFile(imageUrls[0], imagePath);
          if (!downloaded) {
            imageReplyText = `Found a reference image for "${subject}" but couldn't download it to work with.`;
          } else {
            const meshResult = await generateMeshFromImage(imagePath);
            if (!meshResult.success) {
              imageReplyText = `Generating a mesh for "${subject}" failed: ${meshResult.error}`;
            } else {
              let meshPathForImport = meshResult.meshPath!;
              let scaleNote = "Heads up: TripoSR's output isn't real-world scaled, so it may look tiny or huge at first — Fusion's Scale tool can fix that, and geometry detail is rough compared to a hand-modeled asset.";
              if (targetSizeCm !== null) {
                const scaledPath = await scaleMeshToTargetSize(meshResult.meshPath!, targetSizeCm);
                if (scaledPath) {
                  meshPathForImport = scaledPath;
                  scaleNote = `Scaled it so its largest dimension is about ${targetSizeCm}cm — since I can't know which axis TripoSR treats as "height", that's an overall-size match rather than a guaranteed exact height.`;
                } else {
                  scaleNote = `Tried to scale it to ${targetSizeCm}cm but the scaling step failed, so it's still at TripoSR's raw output size — Fusion's Scale tool can fix that manually.`;
                }
              }
              const importResult = await importMeshToFusion(meshPathForImport);
              imageReplyText = importResult.success
                ? `Done — imported a 3D model of "${subject}" into Fusion 360, built from this reference image: ${imageUrls[0]}\n\n${scaleNote}`
                : `Fusion rejected importing the mesh: ${importResult.error}`;
            }
          }
        }
      }

      conversationHistory.push(`Assistant: ${imageReplyText}`);
      saveHistory();
      clearTimeout(timeout);

      return res.json({ response: imageReplyText, hasProposedChanges: false });
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