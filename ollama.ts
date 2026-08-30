// Standalone Ollama call usable outside the /api/chat request cycle — the
// cascade's own callOllama() (nested inside that handler) captures a
// request-scoped AbortController/selectedModel, so it can't be called from
// an independent REST route or a peer module. Split into its own leaf file
// (not left in server.ts) so agent.ts can use it too without creating a
// circular import: server.ts needs to import agent.ts's exports for its
// /api/agent/* routes, so agent.ts importing this back from server.ts would
// deadlock CommonJS's module-loading order (server.ts's top-of-file
// `require('./agent')` would run before server.ts's own definition of this
// function executed, so agent.ts would see it as undefined). A shared leaf
// both sides import from avoids that, same reasoning as search.ts.

const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434/api/generate";

export async function callOllamaModel(prompt: string, numPredict: number, model: string, timeoutMs = 60000): Promise<string> {
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
