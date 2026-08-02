// Runs automatically before `npm run dev` / `npm start` (see package.json's
// "predev"/"prestart" hooks) so web search is ready by the time N.O.A.H
// answers its first message, instead of silently failing until someone
// remembers to start Docker Desktop and the searxng container by hand.
//
// Designed to never block the app from starting: any failure here just logs
// a warning and lets `npm run dev` continue — webSearch() in server.ts
// already degrades gracefully (empty results) when SearXNG isn't reachable.

import { spawnSync, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

const CONTAINER_NAME = "searxng";
const PROJECT_ROOT = path.resolve(__dirname, "..");
const CONFIG_DIR = path.join(PROJECT_ROOT, "searxng");

const DOCKER_CLI_CANDIDATES = [
  "docker",
  path.join(process.env.LOCALAPPDATA ?? "", "Programs", "DockerDesktop", "resources", "bin", "docker.exe"),
  "C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe"
];

const DOCKER_DESKTOP_EXE_CANDIDATES = [
  path.join(process.env.LOCALAPPDATA ?? "", "Programs", "DockerDesktop", "Docker Desktop.exe"),
  "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"
];

function run(cli: string, args: string[]) {
  const result = spawnSync(cli, args, { encoding: "utf8" });
  return { ok: result.status === 0, stdout: (result.stdout ?? "").trim(), stderr: (result.stderr ?? "").trim() };
}

function findWorkingDockerCli(): string | null {
  for (const candidate of DOCKER_CLI_CANDIDATES) {
    if (spawnSync(candidate, ["--version"]).status === 0) return candidate;
  }
  return null;
}

function isDockerEngineUp(dockerCli: string): boolean {
  return run(dockerCli, ["info"]).ok;
}

async function waitUntil(conditionFn: () => boolean, timeoutMs: number, intervalMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (conditionFn()) return true;
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}

async function main() {
  const dockerCli = findWorkingDockerCli();
  if (!dockerCli) {
    console.warn("[searxng] Docker isn't installed (or not on PATH) — skipping web-search startup. N.O.A.H will run fine, but web search will report no results until Docker + the searxng container are set up.");
    return;
  }

  if (!isDockerEngineUp(dockerCli)) {
    console.log("[searxng] Docker engine isn't running — launching Docker Desktop...");
    const desktopExe = DOCKER_DESKTOP_EXE_CANDIDATES.find(p => fs.existsSync(p));
    if (desktopExe) {
      spawn(desktopExe, [], { detached: true, stdio: "ignore" }).unref();
    } else {
      console.warn("[searxng] Couldn't find Docker Desktop.exe — start it manually if web search doesn't work.");
    }

    console.log("[searxng] Waiting up to 90s for the Docker engine to come up (first boot after a restart can be slow)...");
    const ready = await waitUntil(() => isDockerEngineUp(dockerCli), 90000, 3000);
    if (!ready) {
      console.warn("[searxng] Docker engine didn't come up in time — continuing without web search. Once Docker Desktop finishes starting, restart the searxng container manually or just restart N.O.A.H.");
      return;
    }
    console.log("[searxng] Docker engine is up.");
  }

  const inspect = run(dockerCli, ["inspect", "-f", "{{.State.Running}}", CONTAINER_NAME]);

  if (inspect.ok && inspect.stdout === "true") {
    console.log("[searxng] Container already running.");
    return;
  }

  if (inspect.ok && inspect.stdout === "false") {
    console.log("[searxng] Container exists but is stopped — starting it...");
    const started = run(dockerCli, ["start", CONTAINER_NAME]);
    console.log(started.ok ? "[searxng] Started." : `[searxng] Failed to start: ${started.stderr}`);
    return;
  }

  console.log("[searxng] Container not found — creating it for the first time...");
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  const created = run(dockerCli, [
    "run", "-d",
    "--name", CONTAINER_NAME,
    "-p", "8080:8080",
    "-v", `${CONFIG_DIR}:/etc/searxng`,
    "searxng/searxng:latest"
  ]);

  if (!created.ok) {
    console.warn(`[searxng] Failed to create container: ${created.stderr}`);
    return;
  }

  console.log("[searxng] Container created. If searxng/settings.yml doesn't already have `json` under `search.formats`, add it and run `docker restart searxng` once — otherwise the search API will 403.");
}

main().catch(err => {
  console.error("[searxng] Unexpected error during startup check (continuing anyway):", err);
});
