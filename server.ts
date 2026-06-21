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
    runGitCommand(["branch", "-M", "master"]);
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

  const wantsModification =/(modify|change|rewrite|update|edit|improve|refactor)/i.test(message);

const lowerMessage = message.toLowerCase();

  const modificationTarget = lowerMessage.includes("memory") ? "memory.json": lowerMessage.includes("server") || lowerMessage.includes("source code") || lowerMessage.includes("implementation") ? "server.ts": "personality.txt";
  
  const wantsCodeModification = modificationTarget === "server.ts";

  const selfCode = wantsCodeModification ? loadFile(CODE_FILE): "";

  const sourceCodeContext = selfCode ? `--- CURRENT SOURCE CODE (${CODE_FILE}) ---\n${selfCode}\n\n`: "";

  const metaSystemInstruction = (
    `--- CURRENT PERSONALITY ---\n${personality}\n\n` +
    `The text above is the complete contents of personality.txt.\n` +
    `When modifying personality.txt, preserve all existing instructions unless the user explicitly asks to remove them.\n\n` +
    `--- CURRENT MEMORY ---\n${memory}\n\n` +
    `The information in CURRENT MEMORY contains persistent facts and should be treated as true unless the user explicitly corrects them.\n\n` +
    sourceCodeContext +

    `You are N.O.A.H., a self-modifying assistant.\n\n` +
    `The user's requested modification target is: ${modificationTarget}\n\n` +
    `If the user requests a modification to personality.txt, memory.json, or server.ts, you MUST respond using the exact format below.\n` +
    `Any response that does not use this format is invalid.\n\n` +

    `[UPDATE: filename]\n` +
    `\`\`\`language\n` +
    `complete file contents\n` +
    `\`\`\`\n\n` +

    `Rules for modifications:\n` +
    `- Output a complete replacement file.\n` +
    `- Do not output partial snippets.\n` +
    `- Do not use placeholders.\n` +
    `- Do not write 'existing code', 'existing imports', or 'omitted for brevity'.\n` +
    `- Preserve existing functionality unless explicitly instructed otherwise.\n` +
    `- When modifying a file, output ONLY UPDATE blocks and nothing else.\n\n`
  
  );

  try {
    const controller = new AbortController();

    const timeout = setTimeout(() => {
      controller.abort();
    }, 60000);

    const recentHistory = conversationHistory.slice(-20).join("\n");

    const fullPrompt = wantsModification ? `System Instruction:\n${metaSystemInstruction}\n\n` + `User Request:\n${message}`: `System Instruction:\n${metaSystemInstruction}\n\n` + `Conversation History:\n${recentHistory}\n\n` + `User Request:\n${message}`;

    console.log("========== PROMPT ==========");
    console.log(fullPrompt);
    console.log("============================");

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
    
    if (!aiResponse.includes("I cannot execute") && !aiResponse.includes("I am an AI model")) {

      const isRefusal = aiResponse.includes("cannot execute") || aiResponse.includes("cannot modify") || aiResponse.includes("I am an AI model");

      if (!isRefusal) {
        const historySafeResponse = aiResponse.replace(
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

    // Parse out potential updates
    const pattern = /\[UPDATE:\s*([\w\.-]+)\]\s*```[\w]*\n([\s\S]*?)```/g;
    const updates: { filepath: string; content: string }[] = [];
    let match;

    while ((match = pattern.exec(aiResponse)) !== null) {
      updates.push({
        filepath: match[1].trim(),
        content: match[2].trim()
      });
    }

    console.log("RAW AI RESPONSE:");
    console.log(aiResponse);

    console.log("UPDATES FOUND:", updates.length);

    if (updates.length > 0) {
      console.log("FILE:", updates[0].filepath);
      console.log("CONTENT:");
      console.log(updates[0].content);
    }

    let hasProposedChanges = false;
    let gitDiff = "";

    if (updates.length > 0) {
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
      runGitCommand(["checkout", "master"]);
      runGitCommand(["branch", "-D", "feature/ai-web-modification"]);
      pendingFiles = [];
      return res.status(422).json({ error: "TypeScript type checking failed. Merging was aborted for safety." });
    }
  }

  // Commit and merge sequence
  runGitCommand(["add", ...pendingFiles]);
  runGitCommand(["commit", "-m", "AI self-modification merge"]);
  runGitCommand(["checkout", "master"]);

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

  res.json({
    success: true,
    message: "Changes discarded successfully."
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Assistant server running at http://localhost:${PORT}`);
});