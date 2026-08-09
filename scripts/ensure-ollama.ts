// Runs automatically before `npm run dev` / `npm start` (see package.json's
// "predev"/"prestart" hooks) so chat/search/recall don't silently fail with
// "fetch failed" until someone notices Ollama isn't running.
//
// Ollama's Windows tray app ("ollama app.exe", launched by its Startup-folder
// shortcut) has a recurring bug where it crashes immediately after login —
// its own logs show "Failed to start: Unable to set icon: The operation
// completed successfully." (a benign Windows API result its error handling
// mis-treats as fatal) — so the auto-start you'd expect from a reboot often
// silently doesn't happen. This bypasses that entirely by launching the
// `ollama serve` CLI directly, headless, with no tray icon to crash on.
//
// Designed to never block the app from starting: any failure here just logs
// a warning and lets `npm run dev` continue — server.ts's Ollama calls will
// surface their own "fetch failed" error per-request if it's still down.

import { spawnSync, spawn } from "child_process";
import * as path from "path";

const OLLAMA_ORIGIN = (process.env.OLLAMA_URL ?? "http://localhost:11434/api/generate").replace(/\/api\/generate$/, "");

const OLLAMA_CLI_CANDIDATES = [
  "ollama",
  path.join(process.env.LOCALAPPDATA ?? "", "Programs", "Ollama", "ollama.exe")
];

function findWorkingOllamaCli(): string | null {
  for (const candidate of OLLAMA_CLI_CANDIDATES) {
    if (spawnSync(candidate, ["--version"]).status === 0) return candidate;
  }
  return null;
}

async function isOllamaUp(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_ORIGIN}/api/version`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function waitUntil(conditionFn: () => Promise<boolean>, timeoutMs: number, intervalMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await conditionFn()) return true;
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}

async function main() {
  if (await isOllamaUp()) {
    console.log("[ollama] Already running.");
    return;
  }

  const ollamaCli = findWorkingOllamaCli();
  if (!ollamaCli) {
    console.warn("[ollama] Ollama isn't installed (or not on PATH) — skipping startup check. Chat, search, and recall will fail until it's running.");
    return;
  }

  console.log("[ollama] Not running — starting `ollama serve` directly (bypassing the tray app's known crash-on-launch bug)...");
  const proc = spawn(ollamaCli, ["serve"], { detached: true, stdio: "ignore" });
  proc.unref();

  const ready = await waitUntil(isOllamaUp, 30000, 1000);
  console.log(ready ? "[ollama] Up and responding." : "[ollama] Didn't come up in time — continuing anyway, it may just need more time to load.");
}

main().catch(err => {
  console.error("[ollama] Unexpected error during startup check (continuing anyway):", err);
});
