// Agent mode: give Noah a goal, it plans + executes multiple steps toward
// it somewhat independently, using the same tag-based action convention
// already proven for web search and MCP tools in /api/chat ([SEARCH: ...],
// [MCP_TOOL: ...]) — generalized here from "one follow-up, then stop" into
// a bounded adaptive loop (decide next action, execute it, observe the
// real result, decide the next one, ...) rather than a rigid pre-committed
// script, since real execution often needs to adapt based on what's
// actually found.
//
// This whole codebase is synchronous/single-flight — one HTTP request can
// already legitimately block for up to 45 minutes (multiview mesh
// generation) under the shared requestInFlight lock, and there is no
// WebSocket/SSE/job-queue anywhere in this app. Agent mode follows that
// same shape rather than inventing new background-job infrastructure. A
// direct implication: once a run starts, there is no way to interrupt it
// mid-flight (an incoming request would just get 429'd by requestInFlight,
// same as during mesh generation today) — human oversight has to happen
// before the run starts (the plan preview), not during it.
//
// A standalone module, not a refactor of /api/chat's own tag-handling
// block — that code has been hand-tuned through several real bugs this
// session (routing, disambiguation, confident framing) and touching it for
// this carries real regression risk for no benefit. A small amount of
// duplicated tag-detection regex here is a much cheaper price than that.
//
// webSearch lives in its own leaf module (search.ts), not server.ts — this
// file needs it, and server.ts needs to import THIS file's exports for its
// new /api/agent/* routes, so importing webSearch from server.ts directly
// would create a circular require: server.ts's top-of-file `require('./agent')`
// would run before server.ts's own webSearch definition executed, so it
// would come back undefined. A shared leaf module both sides import from
// avoids that, same as mcp.ts/telegram.ts already being leaves server.ts
// imports from, never the reverse.

import { getMcpToolsDescription, callMcpTool, hasMcpTools } from './mcp';
import { webSearch, WebSearchResult } from './search';
import { callOllamaModel } from './ollama';

const CODE_MODEL = process.env.OLLAMA_CODE_MODEL ?? "qwen3.5:9b";
const MAX_STEPS = 8;

const SEARCH_TAG_PATTERN = /\[SEARCH:\s*([^\]]+)\]/i;
const MCP_TOOL_TAG_PATTERN = /\[MCP_TOOL:\s*([^\]]+)\]/i;

export interface AgentStep {
  action: string;
  result: string;
}

export interface AgentRunResult {
  finalAnswer: string;
  steps: AgentStep[];
  hitStepLimit: boolean;
}

interface AgentProgress {
  active: boolean;
  currentStep: number;
  maxSteps: number;
  description: string;
}

// Module-level, same shape as server.ts's gpuExclusiveTaskRunning — read by
// GET /api/hud-metrics (already polled every 3s by the frontend, already
// piggybacking one other async-originated field this way, the vision
// proactiveGreeting) so the panel can show live status with zero new
// transport.
let progress: AgentProgress = { active: false, currentStep: 0, maxSteps: MAX_STEPS, description: "" };

export function getAgentProgress(): AgentProgress {
  return progress;
}

function formatToolResultForPrompt(searchResults: WebSearchResult[]): string {
  return searchResults.length > 0
    ? searchResults.map((r, i) => `${i + 1}. ${r.title}\n${r.description}\n${r.url}`).join("\n\n")
    : "(no results found)";
}

// Informational only — a rough, human-readable preview for the user to
// look over before committing to an unsupervised run, not a script the
// real run has to follow. Deliberately plain text, not JSON: this doesn't
// need the reliability the actual tag-based execution does, so it doesn't
// need that risk either.
export async function previewAgentPlan(goal: string): Promise<string> {
  const toolsBlock = getMcpToolsDescription();
  const prompt = `You are about to work on this goal somewhat independently, taking multiple steps if needed: "${goal}"

You have web search available, ${hasMcpTools() ? "plus these external tools:\n" + toolsBlock : "and no external tools are currently connected."}

Before starting, describe your rough intended approach as a short numbered list (3-6 steps max). This is a preview for a human to review, not a commitment to exact steps — the real execution may adapt based on what it actually finds. Reply with ONLY the numbered list, no other text.`;

  try {
    const raw = await callOllamaModel(prompt, 400, CODE_MODEL, 60000);
    return raw.trim();
  } catch (err: any) {
    return `Couldn't generate a preview: ${err.message}`;
  }
}

// The real, adaptive execution. Runs to completion unsupervised once
// called — see the file header for why that's an accepted constraint here,
// not an oversight.
export async function runAgent(goal: string): Promise<AgentRunResult> {
  const toolsBlock = getMcpToolsDescription();
  const steps: AgentStep[] = [];
  progress = { active: true, currentStep: 0, maxSteps: MAX_STEPS, description: `Starting: ${goal}` };

  try {
    for (let i = 0; i < MAX_STEPS; i++) {
      progress = { active: true, currentStep: i + 1, maxSteps: MAX_STEPS, description: i === 0 ? "Thinking about the first step..." : "Deciding the next step..." };

      const stepsSoFar = steps.length > 0
        ? "Steps taken so far:\n" + steps.map((s, idx) => `Step ${idx + 1} — Action: ${s.action}\nResult: ${s.result}`).join("\n\n") + "\n\n"
        : "";

      const loopPrompt = `You are an autonomous agent working step by step toward this goal: "${goal}"

You have web search available, ${hasMcpTools() ? "plus these external tools:\n" + toolsBlock : "and no external tools are currently connected."}

${stepsSoFar}To take an action, respond with EXACTLY one of these and nothing else:

[SEARCH: concise web search query]

or

[MCP_TOOL: qualifiedName]
{"argument": "value"}

If you already have enough information from the steps above to fully address the goal, respond with your final answer directly in plain prose instead — no tag, no action, just the answer. Only take an action if you genuinely still need more information.`;

      const response = await callOllamaModel(loopPrompt, 800, CODE_MODEL, 90000);

      const mcpMatch = response.match(MCP_TOOL_TAG_PATTERN);
      const searchMatch = !mcpMatch ? response.match(SEARCH_TAG_PATTERN) : null;

      if (mcpMatch) {
        const qualifiedName = mcpMatch[1].trim();
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        let toolArgs: Record<string, unknown> = {};
        if (jsonMatch) {
          try { toolArgs = JSON.parse(jsonMatch[0]); } catch { /* call with no args */ }
        }
        progress = { active: true, currentStep: i + 1, maxSteps: MAX_STEPS, description: `Calling tool: ${qualifiedName}` };
        let result: string;
        try {
          result = await callMcpTool(qualifiedName, toolArgs);
        } catch (err: any) {
          result = `Tool call failed: ${err.message}`;
        }
        steps.push({ action: `[MCP_TOOL: ${qualifiedName}]`, result });
        continue;
      }

      if (searchMatch) {
        const query = searchMatch[1].trim();
        progress = { active: true, currentStep: i + 1, maxSteps: MAX_STEPS, description: `Searching: ${query}` };
        const results = await webSearch(query);
        steps.push({ action: `[SEARCH: ${query}]`, result: formatToolResultForPrompt(results) });
        continue;
      }

      // No action tag — the model is giving its final answer.
      return { finalAnswer: response.trim(), steps, hitStepLimit: false };
    }

    // Loop exhausted without a final answer — return what was gathered
    // honestly rather than pretending the goal was fully completed.
    const lastResult = steps.length > 0 ? steps[steps.length - 1].result : "";
    return {
      finalAnswer: `I reached my step limit (${MAX_STEPS}) before fully completing this. Here's what I found so far:\n\n${lastResult}`,
      steps,
      hitStepLimit: true
    };
  } finally {
    progress = { active: false, currentStep: 0, maxSteps: MAX_STEPS, description: "" };
  }
}
