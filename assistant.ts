import * as fs from 'fs';
import { execSync } from 'child_process';
import * as readline from 'readline/promises';
import { stdin as input, stdout as output } from 'process';

const OLLAMA_URL = "http://localhost:11434/api/generate";
const MODEL_NAME = "qwen2.5-coder:7b"; // Ensure you have this pulled in Ollama

const PERSONALITY_FILE = "personality.txt";
const CODE_FILE = "assistant.ts";

function loadFile(filepath: string): string {
  try {
    return fs.readFileSync(filepath, 'utf8');
  } catch {
    return "";
  }
}

function writeFile(filepath: string, content: string): void {
  fs.writeFileSync(filepath, content, 'utf8');
}

function runGitCommand(args: string[]): string {
  try {
    return execSync(`git ${args.join(' ')}`, { stdio: 'pipe' }).toString().trim();
  } catch (error: any) {
    // If command fails, return the error message
    return error.stderr?.toString().trim() || error.message;
  }
}

function setupGitBranches(): void {
  const status = runGitCommand(["status", "--porcelain"]);
  if (status) {
    console.log("\n⚠️ Warning: You have uncommitted changes. Please commit or stash them first.");
    process.exit(1);
  }

  const currentBranch = runGitCommand(["branch", "--show-current"]);
  if (!currentBranch) {
    // Initialize repository with a commit if empty
    runGitCommand(["add", "."]);
    runGitCommand(["commit", "-m", "Initial commit"]);
    runGitCommand(["branch", "-M", "develop"]);
  }
}

async function askLocalLLM(prompt: string, systemPrompt: string): Promise<string> {
  const payload = {
    model: MODEL_NAME,
    prompt: `System Instruction:\n${systemPrompt}\n\nUser Request:\n${prompt}`,
    stream: false
  };

  try {
    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as { response: string };
    return data.response;
  } catch (error) {
    console.error("Error communicating with local LLM:", error);
    process.exit(1);
  }
}

interface FileUpdate {
  filepath: string;
  content: string;
}

function parseProposedChanges(aiResponse: string): FileUpdate[] {
  // Regex matches pattern: [UPDATE: filename] followed by markdown code blocks
  const pattern = /\[UPDATE:\s*([\w\.-]+)\]\s*```[\w]*\n([\s\S]*?)```/g;
  const updates: FileUpdate[] = [];
  let match;

  while ((match = pattern.exec(aiResponse)) !== null) {
    updates.push({
      filepath: match[1].trim(),
      content: match[2].trim()
    });
  }

  return updates;
}

async function proposeAndMergeFlow(fileUpdates: FileUpdate[]): Promise<void> {
  console.log("\n--- Proposed Changes Detected ---");

  // Create and switch to temporary feature branch
  runGitCommand(["checkout", "-b", "feature/ai-self-modification"]);

  const modifiedFiles: string[] = [];

  for (const update of fileUpdates) {
    const allowedFiles = [PERSONALITY_FILE, CODE_FILE];
    if (!allowedFiles.includes(update.filepath)) {
      console.log(`Skipping unauthorized file modification request for: ${update.filepath}`);
      continue;
    }

    writeFile(update.filepath, update.content);
    modifiedFiles.push(update.filepath);
    console.log(`✓ Drafted changes to ${update.filepath}`);
  }

  // Display Git Diff
  console.log("\n=== GIT DIFF ===");
  console.log(runGitCommand(["diff"]));
  console.log("================");

  // Human-in-the-loop approval
  const rl = readline.createInterface({ input, output });
  const approval = await rl.question("\nDo you want to merge these changes into 'develop'? (y/N): ");
  rl.close();

  if (approval.trim().toLowerCase() === "y") {
    // Perform TypeScript type check if codebase was modified
    if (modifiedFiles.includes(CODE_FILE)) {
      console.log("Running TypeScript compilation check...");
      try {
        // Run TypeScript compiler check without generating output files
        execSync(`npx tsc --noEmit`, { stdio: 'pipe' });
        console.log("✓ TypeScript type check passed safely.");
      } catch (error) {
        console.log("❌ Compilation check failed! Reverting proposed modifications.");
        runGitCommand(["checkout", "develop"]);
        runGitCommand(["branch", "-D", "feature/ai-self-modification"]);
        return;
      }
    }

    // Commit changes to feature branch
    runGitCommand(["add", ...modifiedFiles]);
    runGitCommand(["commit", "-m", "AI suggested self-modification"]);

    // Merge to develop branch
    runGitCommand(["checkout", "develop"]);
    runGitCommand(["merge", "feature/ai-self-modification"]);
    runGitCommand(["branch", "-D", "feature/ai-self-modification"]);

    console.log("\n🎉 Changes successfully merged to 'develop'!");
    console.log("Please restart the script to apply any modifications made to assistant.ts.");
  } else {
    // Revert changes
    console.log("\n❌ Changes rejected. Reverting to 'develop'...");
    runGitCommand(["checkout", "develop"]);
    runGitCommand(["branch", "-D", "feature/ai-self-modification"]);
  }
}

async function main() {
  setupGitBranches();

  const personality = loadFile(PERSONALITY_FILE);
  const selfCode = loadFile(CODE_FILE);

  console.log(`Assistant running with model '${MODEL_NAME}' on branch 'develop'.`);
  console.log("You can ask normal questions, or request changes to my personality or code.");
  console.log("Type 'exit' to quit.\n");

  const metaSystemInstruction = (
    `${personality}\n\n` +
    `You are a self-modifying assistant. Your current source code is provided below.\n` +
    `--- CURRENT SOURCE CODE (${CODE_FILE}) ---\n${selfCode}\n` +
    `--- CURRENT PERSONALITY ({PERSONALITY_FILE}) ---\n${personality}\n\n` +
    `If the user asks you to change your personality or code, formulate the changes. ` +
    `To update a file, output your proposed replacement strictly in the following format:\n` +
    `[UPDATE: filename]\n` +
    `\`\`\`language\n` +
    `complete content of the file goes here\n` +
    `\`\`\`\n` +
    `Make sure to replace the entire file contents when proposing an update. ` +
    `Do not write conversational text inside the markdown block.`
  );

  const rl = readline.createInterface({ input, output });

  while (true) {
    try {
      const userInput = await rl.question("\nYou: ");
      const trimmedInput = userInput.trim();

      if (!trimmedInput) continue;
      if (trimmedInput.toLowerCase() === "exit" || trimmedInput.toLowerCase() === "quit") {
        break;
      }

      console.log("Thinking...");
      const response = await askLocalLLM(trimmedInput, metaSystemInstruction);

      const updates = parseProposedChanges(response);

      if (updates.length > 0) {
        // Strip the block formatting before displaying conversational output to the terminal
        const conversationalText = response.replace(/\[UPDATE:.*?\]\s*```[\s\S]*?```/g, "").trim();
        if (conversationalText) {
          console.log(`\nAssistant: ${conversationalText}`);
        }
        
        // Close readline interface temporarily to hand control over to the sub-process
        rl.close();
        await proposeAndMergeFlow(updates);
        // Re-open readline for the loop
        return main(); 
      } else {
        console.log(`\nAssistant: ${response}`);
      }

    } catch (error) {
      console.error("\nAn error occurred in the loop:", error);
      break;
    }
  }
  rl.close();
}

main();