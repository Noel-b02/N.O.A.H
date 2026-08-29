# Telegram bot setup (one-time)

The Telegram bridge lets you text or send voice notes to Noah from anywhere — not just the local web UI — and get text + voice-note replies back. It reuses Noah's real chat handler, so it can do everything the web UI can: image generation/editing, face enrollment, memory, document Q&A, all of it.

No public internet exposure is needed for this. Unlike a phone-call or WhatsApp integration, Telegram bots work by **long-polling** — Noah's server reaches out to Telegram, not the other way around — so nothing needs to be reachable from outside your network.

## 1. Create the bot

1. Open Telegram and start a chat with **@BotFather** (Telegram's own official bot for creating bots).
2. Send `/newbot`, then follow its prompts — pick a display name and a username ending in `bot` (e.g. `NoahAssistantBot`).
3. BotFather replies with a token that looks like `123456789:AAH...`. Put it in `.env`:

   ```
   TELEGRAM_BOT_TOKEN=123456789:AAH...
   ```

4. Restart Noah (`npm run dev`). You should see `[telegram] Starting long-poll loop...` in the console.

## 2. Claim the bot (authorize your account)

This step exists so a stranger who finds your bot's username can never talk to Noah — only one specific Telegram account, chosen by you, ever gets a real reply.

1. In Telegram, find your bot (search its username) and send it any message — "hi" is fine.
2. It will reply with your Telegram chat ID and these exact instructions:
   ```
   TELEGRAM_AUTHORIZED_CHAT_ID=<your id>
   ```
3. Add that line to `.env` and restart Noah.

Until this step is done, the bot's only behavior for *anyone* who messages it is handing back their chat ID — it never reaches Noah's actual chat handler. Once set, any message from a different chat ID is silently ignored (no reply at all), so nobody else can tell the bot is even doing anything.

## Using it

Text or send a voice note like you would in the web UI. Every reply comes back as both a text message and a voice note. Attach a photo (with or without a caption) the same way you'd attach one in the web UI — "remember this face as X", "make this look like a watercolor painting", or just send a photo with no caption for a plain description.

## What it doesn't do (yet)

- **No proactive/unprompted messages** — Noah won't message you first (e.g. to alert you about something the vision system noticed) unless a future feature specifically wires that up. Today it only replies to messages you send.
- **Messages sent while Noah's server was offline are not answered late** — they're deliberately dropped on the next startup rather than replayed, so a command like a print confirmation can't fire hours after you meant it to. Send it again once Noah's back up.
- **Only one authorized account.** If you ever want a second person (e.g. a family member) to be able to message the bot too, that needs a small code change — it's a single chat ID today, not a list.
