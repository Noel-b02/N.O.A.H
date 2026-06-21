import express from 'express';
import * as fs from 'fs';
import { execSync } from 'child_process';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434/api/generate";

const MODEL_NAME = process.env.OLLAMA_MODEL ?? "qwen3.5:4b";
const PERSONALITY_FILE = "personality.txt";
const CODE_FILE = "server.ts"; // The assistant can modify this file (itself)
const MEMORY_FILE = "memories/memory.json";
const HISTORY_FILE = "memories/history/active.json";

// State to track if we have a draft on a feature branch
let pendingFiles: string[] = [];
let activeDraftBranch: string | null = null;
let conversationHistory: string[] = loadHistory();


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
    return execSync(`git ${args.join(' ')}`, { stdio: 'pipe' }).toString().trim();
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
    runGitCommand(["branch", "-M", "develop"]);
  }
};
setupGit();

app.get('/api/status', (_, res) => {
  res.json({
    activeDraftBranch,
    pendingFiles,
    historySize: conversationHistory.length
  });
});

// API: Send chat prompt to the assistant
app.post('/api/chat', async (req, res) => {
 const { message } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message is required." });
  }

  conversationHistory.push(`User: ${message}`);
  saveHistory();

  const personality = loadFile(PERSONALITY_FILE);
  const memory = loadMemory();

  const wantsModification = /(modify|change|rewrite|update|edit|improve|refactor)/i.test(message);

  const selfCode = wantsModification ? loadFile(CODE_FILE): "";

  const sourceCodeContext = selfCode ? `--- CURRENT SOURCE CODE (${CODE_FILE}) ---\n${selfCode}\n\n`: "";

  const modificationInstructions = wantsModification ? `
  If the user requests a modification, you MUST respond using:

  [UPDATE: filename]
  \`\`\`
  complete file contents
  \`\`\`

  Only provide complete replacements.
  Do not use placeholders.
  Do not omit code.
  `
  : "";

  const metaSystemInstruction = (
    `--- CURRENT PERSONALITY (${PERSONALITY_FILE}) ---\n${personality}\n\n` +
    `--- CURRENT MEMORY (${MEMORY_FILE}) ---\n${memory}\n\n` +
    sourceCodeContext +
    modificationInstructions +
    `You are a self-modifying assistant. Your current source code is provided above.\n` +
    `You must preserve all existing functionality unless the user explicitly requests its removal.\n` +
    `You must never replace code with placeholders such as:\n` +
    `// existing code\n` +
    `// existing imports\n` +
    `omitted for brevity\n\n` +
    `If the user asks you to change your personality or code, formulate the changes.\n` +
    `To update a file, output your proposed replacement strictly in the following format:\n` +
    `[UPDATE: filename]\n` +
    `\`\`\`language\n` +
    `complete content of the file goes here\n` +
    `\`\`\`\n` +
    `The replacement must be complete, runnable and self-contained.\n` +
    `Make sure to replace the entire file contents when proposing an update.\n` +
    `Do not write conversational text inside the markdown block.`
  );

  try {
    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, 60000);

    const recentHistory = conversationHistory.slice(-20).join("\n");

    const fullPrompt = `System Instruction:\n${metaSystemInstruction}\n\n` + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;

    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL_NAME,
        prompt: fullPrompt,
        stream: false,
        think: false,
        options: {
          num_predict: 150,
          temperature: 0.3
        }
      })
    });

    if (!response.ok) throw new Error("Ollama connection failed");
    const data = await response.json() as { response: string };
    
    clearTimeout(timeout);
    
    const aiResponse = data.response;
    
    conversationHistory.push(`Assistant: ${aiResponse}`);
    saveHistory();

    if (conversationHistory.length > 40) {
      conversationHistory =
      conversationHistory.slice(-40);
  }

    // Parse out potential updates
    const pattern = /\[UPDATE:\s*([\w\.-]+)\]\s*```[\w]*\n([\s\S]*?)```/g;
    const updates: { filepath: string; content: string }[] = [];
    let match;

    while ((match = pattern.exec(aiResponse)) !== null) {
      updates.push({ filepath: match[1].trim(), content: match[2].trim() });
    }

    let hasProposedChanges = false;
    let gitDiff = "";

    if (updates.length > 0) {
      // 1. Ensure clean workspace status before starting a draft branch
      const status = runGitCommand(["status", "--porcelain"]);
      if (status) {
        return res.json({ 
          response: "I attempted to draft changes, but the local workspace has uncommitted files. Please resolve them first.",
          hasProposedChanges: false 
        });
      }

      // 2. Checkout new draft branch
    const branchName =
      `feature/ai-${Date.now()}`;

    runGitCommand([
      "checkout",
      "-b",
      branchName
    ]);

    activeDraftBranch = branchName;      
    pendingFiles = [];

      // 3. Write drafts to disk
      for (const update of updates) {
        if (
          [PERSONALITY_FILE, CODE_FILE, MEMORY_FILE]
            .includes(update.filepath)
        ) {
          writeFile(update.filepath, update.content);
          pendingFiles.push(update.filepath);
        }
      }

      hasProposedChanges = true;
      gitDiff = runGitCommand(["diff"]);
    }

    // Strip update tags out of conversational response text
    const cleanText = aiResponse.replace(/\[UPDATE:.*?\]\s*```[\s\S]*?```/g, "").trim();

    res.json({
      response: cleanText || (hasProposedChanges ? "I have drafted the requested changes for your review." : ""),
      hasProposedChanges,
      diff: gitDiff
    });

  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// API: Approve and Merge
app.post('/api/approve', (req, res) => {
  if (pendingFiles.length === 0) {
    return res.status(400).json({ error: "No pending modifications to approve." });
  }

  // Type-check safety step if the server script was modified
  if (pendingFiles.includes(CODE_FILE)) {
    try {
      execSync("npx tsc --noEmit", { stdio: "pipe" });
    } catch (e: any) {
      // Revert branch upon compilation failure
      runGitCommand(["checkout", "develop"]);
      runGitCommand(["branch", "-D", "feature/ai-web-modification"]);
      pendingFiles = [];
      return res.status(422).json({ error: "TypeScript type checking failed. Merging was aborted for safety." });
    }
  }

  // Commit and merge sequence
  runGitCommand(["add", ...pendingFiles]);
  runGitCommand(["commit", "-m", "AI self-modification merge"]);
  runGitCommand(["checkout", "develop"]);

  if (activeDraftBranch) {
    runGitCommand([
      "merge",
      activeDraftBranch
    ]);

    runGitCommand([
      "branch",
      "-D",
      activeDraftBranch
    ]);
  }

  activeDraftBranch = null;
  pendingFiles = [];
  res.json({ success: true, message: "Changes successfully merged to 'develop'!" });
});

// API: Reject and Discard
app.post('/api/reject', (req, res) => {

  runGitCommand(["checkout", "develop"]);
  if (activeDraftBranch) {
  runGitCommand([
    "branch",
    "-D",
    activeDraftBranch
  ]);
  } 
  activeDraftBranch = null;
  pendingFiles = [];
  res.json({ success: true, message: "Changes discarded successfully." });
});

app.listen(PORT, () => {
  console.log(`🚀 Assistant server running at http://localhost:${PORT}`);
});