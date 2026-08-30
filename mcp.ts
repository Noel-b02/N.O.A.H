// MCP (Model Context Protocol) client support — lets Noah call tools exposed
// by external MCP servers (spawned as local stdio child processes) during
// chat. Tool availability/invocation is surfaced to the LLM via the same
// hand-rolled tag convention server.ts already uses for web search
// ([SEARCH: query]) rather than Ollama's native tool-calling API, since that
// only exists on /api/chat (a messages-array endpoint) while this whole
// codebase is built around /api/generate's single flattened prompt string —
// see the plan notes for the full reasoning. This keeps MCP tool-calling
// model-agnostic (works with any model that can follow a formatting
// instruction) instead of depending on a specific model's tool-calling
// chat template.
//
// Config lives in mcp-servers.json (gitignored — server args/paths are
// machine-specific, same reasoning as .env) with mcp-servers.json.example
// committed as a template. No servers configured = no tools offered, no
// prompt bloat, silently a no-op.

import * as fs from 'fs';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const MCP_CONFIG_FILE = 'mcp-servers.json';

interface McpServerConfig {
  name: string;
  command: string;
  args: string[];
  env?: Record<string, string>;
}

interface McpToolInfo {
  serverName: string;
  toolName: string;
  qualifiedName: string;
  description: string;
  inputSchema: unknown;
}

// serverName -> connected client, used by callMcpTool to dispatch. Separate
// from mcpTools (which is the flat, prompt-friendly list) so a lookup by
// server doesn't require scanning the tool list.
const mcpClients = new Map<string, Client>();
let mcpTools: McpToolInfo[] = [];

function loadMcpServerConfigs(): McpServerConfig[] {
  try {
    return JSON.parse(fs.readFileSync(MCP_CONFIG_FILE, 'utf8'));
  } catch {
    return [];
  }
}

// Tool names are only unique within a server, not across servers — this
// qualified form (server__tool) is what the LLM is actually shown and what
// it must echo back in a [MCP_TOOL: ...] tag, so callMcpTool can always
// resolve it back to the right client unambiguously.
function qualify(serverName: string, toolName: string): string {
  return `${serverName}__${toolName}`;
}

// Connects to every configured server in parallel via Promise.allSettled —
// one bad command/config must not block the others or crash Noah's boot,
// matching the "never fail hard" convention used elsewhere in this project
// (listPulledModels(), isServiceHealthy()).
export async function startMcpServers(): Promise<void> {
  const configs = loadMcpServerConfigs();
  if (configs.length === 0) {
    console.log('[mcp] No servers configured (mcp-servers.json missing or empty) — skipping.');
    return;
  }

  await Promise.allSettled(configs.map(async (config) => {
    try {
      console.log(`[mcp] Connecting to server "${config.name}" (${config.command} ${config.args.join(' ')})...`);

      const transport = new StdioClientTransport({
        command: config.command,
        args: config.args,
        env: config.env
      });

      const client = new Client({ name: 'noah', version: '1.0.0' });
      await client.connect(transport);

      const { tools } = await client.listTools();
      mcpClients.set(config.name, client);
      for (const tool of tools) {
        mcpTools.push({
          serverName: config.name,
          toolName: tool.name,
          qualifiedName: qualify(config.name, tool.name),
          description: tool.description ?? '(no description provided)',
          inputSchema: tool.inputSchema
        });
      }

      console.log(`[mcp] Server "${config.name}" connected — ${tools.length} tool(s): ${tools.map(t => t.name).join(', ') || '(none)'}`);
    } catch (err: any) {
      console.warn(`[mcp] Failed to connect to server "${config.name}": ${err.message}`);
    }
  }));
}

// Fire-and-forget, same reasoning as stopSpeechService()/stopVisionService()
// — called from shutdown()'s synchronous signal handlers, where there's no
// opportunity to await an async cleanup before process.exit(0). close()
// still runs (and kills the underlying child process) even though nothing
// here waits for it to finish.
export function stopMcpServers(): void {
  for (const [name, client] of mcpClients) {
    console.log(`[mcp] Stopping server "${name}"...`);
    client.close().catch(() => { /* best-effort */ });
  }
}

// Used by server.ts to decide whether a plain chat turn should route to the
// bigger/"complex" model — confirmed live that the small default chat model
// (qwen3.5:4b) narrated an intention to use a tool ("let me check...") but
// never actually emitted the [MCP_TOOL: ...] tag, while the same exact
// prompt against the bigger model did it correctly on the first try. Same
// "needs reliable structured output" reasoning this file already applies to
// wantsModification forcing CODE_MODEL regardless of message length.
export function hasMcpTools(): boolean {
  return mcpTools.length > 0;
}

// Empty string when no tools are connected — appended straight into
// server.ts's system-instruction block, so a no-op here means zero prompt
// bloat when MCP isn't configured at all.
export function getMcpToolsDescription(): string {
  if (mcpTools.length === 0) return '';

  // Strips `$schema` (a pure JSON-Schema-spec-version marker, never useful
  // to the model) before stringifying — with 10+ tools in a typical MCP
  // server this measurably shortens the prompt for no loss of information.
  const toolList = mcpTools.map(t => {
    const { $schema, ...schema } = (t.inputSchema ?? {}) as Record<string, unknown>;
    return `- ${t.qualifiedName}: ${t.description}\n  Arguments schema: ${JSON.stringify(schema)}`;
  }).join('\n');

  return `
You also have access to the following external tools. These reach LOCAL resources (files, data, devices) that a web search cannot — if the request is about something one of these tools covers (e.g. a local file's contents), use a tool, never [SEARCH: ...], since a web search will never find local file contents. Use a tool ONLY when it's genuinely necessary to answer the user's request — not for things you already know or can answer directly. To call one, respond with EXACTLY this and nothing else, no other words:

[MCP_TOOL: qualifiedName]
{"argument": "value"}

The JSON object on the second line must match the tool's arguments schema. Use {} if it takes no arguments.

Available tools:
${toolList}
`;
}

export async function callMcpTool(qualifiedName: string, args: Record<string, unknown>): Promise<string> {
  const tool = mcpTools.find(t => t.qualifiedName === qualifiedName);
  if (!tool) {
    throw new Error(`No connected tool named "${qualifiedName}". Available: ${mcpTools.map(t => t.qualifiedName).join(', ') || '(none)'}`);
  }

  const client = mcpClients.get(tool.serverName);
  if (!client) {
    throw new Error(`Tool "${qualifiedName}" resolved to server "${tool.serverName}", but it isn't connected.`);
  }

  const result = await client.callTool({ name: tool.toolName, arguments: args });
  const content = result.content as { type: string; text?: string }[];
  return content
    .filter(c => c.type === 'text' && c.text)
    .map(c => c.text)
    .join('\n') || '(tool returned no text content)';
}
