import os
import subprocess
import sys
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:7b" # Or "llama3", or another model you have pulled in Ollama

PERSONALITY_FILE = "personality.txt"
CODE_FILE = "assistant.py"

def load_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def run_git_command(args):
    """Helper to run git commands and return output."""
    result = subprocess.run(["git"] + args, capture_output=True, text=True)
    return result.stdout.strip()

def setup_git_branches():
    """Ensure we are on a clean develop or feature branch workspace."""
    status = run_git_command(["status", "--porcelain"])
    if status:
        print("\n⚠️ Warning: You have uncommitted changes. Please commit or stash them first.")
        sys.exit(1)
        
    current_branch = run_git_command(["branch", "--show-current"])
    if not current_branch:
        # If git is empty, commit initial files
        run_git_command(["add", "."])
        run_git_command(["commit", "-m", "Initial commit"])
        run_git_command(["branch", "-M", "develop"])

def ask_local_llm(prompt, system_prompt):
    """Sends a request to the local Ollama instance."""
    payload = {
        "model": MODEL_NAME,
        "prompt": f"System Instruction:\n{system_prompt}\n\nUser Request:\n{prompt}",
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error communicating with local LLM: {e}")
        sys.exit(1)

def parse_proposed_changes(ai_response):
    """
    Parses the response to find if the AI proposed code or personality edits.
    Looks for blocks like:
    [UPDATE: filename]
    ```
    content
    ```
    """
    # Regex to extract files and code blocks
    pattern = r"\[UPDATE:\s*([\w\.-]+)\]\s*```[\w]*\n(.*?)```"
    matches = re.findall(pattern, ai_response, re.DOTALL)
    return matches

def propose_and_merge_flow(file_updates):
    """Handles the git branch creation, diffing, and user approval workflow."""
    print("\n--- Proposed Changes Detected ---")
    
    # Switch to a temporary feature branch for safe review
    run_git_command(["checkout", "-b", "feature/ai-self-modification"])
    
    modified_files = []
    for filepath, content in file_updates:
        filepath = filepath.strip()
        # Verify the AI is only editing allowed files for safety
        if filepath not in [PERSONALITY_FILE, CODE_FILE]:
            print(f"Skipping unauthorized file modification request for: {filepath}")
            continue
            
        # Write the new version of the file
        write_file(filepath, content.strip())
        modified_files.append(filepath)
        print(f"✓ Drafted changes to {filepath}")

    # Show the git diff to the user
    print("\n=== GIT DIFF ===")
    print(run_git_command(["diff"]))
    print("================")

    # Human-in-the-loop check
    approval = input("\nDo you want to merge these changes into 'develop'? (y/N): ").strip().lower()

    if approval == "y":
        # Check if python changes compile before merging
        if CODE_FILE in modified_files:
            try:
                subprocess.run([sys.executable, "-m", "py_compile", CODE_FILE], check=True)
                print("✓ Python code compilation check passed.")
            except subprocess.CalledProcessError:
                print("❌ Compilation check failed! Reverting changes.")
                run_git_command(["checkout", "develop"])
                run_git_command(["branch", "-D", "feature/ai-self-modification"])
                return

        # Commit changes on the feature branch
        run_git_command(["add"] + modified_files)
        run_git_command(["commit", "-m", "AI suggested self-modification"])
        
        # Merge back to develop
        run_git_command(["checkout", "develop"])
        run_git_command(["merge", "feature/ai-self-modification"])
        run_git_command(["branch", "-D", "feature/ai-self-modification"])
        print("\n🎉 Changes successfully merged to 'develop'!")
        print("Please restart the script to apply any changes made to assistant.py.")
    else:
        # Revert changes and delete temporary branch
        print("\n❌ Changes rejected. Reverting to 'develop'...")
        run_git_command(["checkout", "develop"])
        run_git_command(["branch", "-D", "feature/ai-self-modification"])

def main():
    setup_git_branches()
    
    personality = load_file(PERSONALITY_FILE)
    self_code = load_file(CODE_FILE)

    print(f"Assistant running with model '{MODEL_NAME}' on branch 'develop'.")
    print("You can ask normal questions, or request changes to my personality or code.")
    print("Type 'exit' to quit.\n")

    # We append self-awareness meta-instructions to the prompt
    meta_system_instruction = (
        f"{personality}\n\n"
        f"You are a self-modifying assistant. Your current source code is provided below.\n"
        f"--- CURRENT SOURCE CODE ({CODE_FILE}) ---\n{self_code}\n"
        f"--- CURRENT PERSONALITY ({PERSONALITY_FILE}) ---\n{personality}\n\n"
        "If the user asks you to change your personality or code, formulate the changes. "
        "To update a file, output your proposed replacement strictly in the following format:\n"
        "[UPDATE: filename]\n"
        "```language\n"
        "complete content of the file goes here\n"
        "```\n"
        "Make sure to replace the entire file contents when proposing an update. "
        "Do not write conversational text inside the markdown block."
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Thinking...")
            response = ask_local_llm(user_input, meta_system_instruction)
            
            # Check if there are updates proposed in the response
            updates = parse_proposed_changes(response)
            
            if updates:
                # Remove the update blocks from conversational output to display explanations clearly
                conversational_text = re.sub(r"\[UPDATE:.*?\]\s*```.*?```", "", response, flags=re.DOTALL).strip()
                if conversational_text:
                    print(f"\nAssistant: {conversational_text}")
                propose_and_merge_flow(updates)
            else:
                print(f"\nAssistant: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()