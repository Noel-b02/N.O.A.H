# MCP server setup (optional)

MCP (Model Context Protocol) lets Noah call tools exposed by external servers — reading local files, or eventually controlling smart-home devices, once you connect a server for that. Nothing is configured by default; this is entirely opt-in.

Noah connects to MCP servers the same way most MCP clients do (Claude Desktop, etc.): each server is a command Noah spawns as a local child process and talks to over stdin/stdout. No public internet exposure, no accounts, no keys — just a program on your own machine.

## 1. Configure a server

Copy the example config:

```
cp mcp-servers.json.example mcp-servers.json
```

Edit `mcp-servers.json` — an array of servers, each with a name, a command, and its arguments:

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/path/to/a/folder/you/want/noah/to/access"]
  }
]
```

`mcp-servers.json` is gitignored (like `.env`) since server paths/commands are specific to your machine.

Restart Noah (`npm run dev`). You should see, for each configured server:

```
[mcp] Connecting to server "filesystem" (...)...
[mcp] Server "filesystem" connected — N tool(s): ...
```

A server that fails to connect (bad command, crashes on start) logs a warning and is skipped — it won't stop Noah from starting or affect any other configured server.

## Using it

Just ask normally — "what's in my notes.txt file", "list what's in that folder". Noah decides on its own whether a request needs a tool; you don't invoke tools by name. If no servers are configured, this is entirely invisible — Noah behaves exactly as it always has.

Tool calls always route through the bigger/"complex" model whenever any MCP server is connected — confirmed live that the smaller default chat model would narrate an intention to use a tool without ever actually calling it, while the bigger model reliably did. This only kicks in once you've configured a server; it has zero effect otherwise.

## What this doesn't do (yet)

- **No smart-home server included.** This is generic MCP client infrastructure, verified against the official filesystem reference server since there's no bundled smart-home integration yet. Any MCP-compliant server (Home Assistant, Hue, etc.) can be added the same way once you have one — it's just another entry in the config, no code changes needed.
- **No tool-use confirmation prompt.** Unlike self-modification (which stages a diff for you to approve) or printing (which asks first), a tool call happens as soon as the model decides to make one. Only point Noah at servers/directories you're comfortable it can read from — and write to, if the server supports writes — without asking first.
- **Read-only by convention, not enforcement.** Whether a tool can write/modify things depends entirely on the server you connect — the official filesystem server, for example, includes `write_file`/`edit_file`/`move_file`. Scope the directory you hand it accordingly.
