# Bambu Lab printer setup (one-time)

Lets Noah take a generated model all the way from "make me a model of X"
through to an actual physical print: repairing/validating the mesh,
slicing it with Bambu Studio, and handing the sliced file off to
[Bambu Connect](https://wiki.bambulab.com/en/software/bambu-connect) —
with your explicit confirmation before anything actually prints. This is
the next stage after the image-to-3D pipeline (`HUNYUAN3D_SETUP.md`) and
Fusion 360 import (`FUSION_SETUP.md`); those two produce and preview the
model, this one gets it onto the bed.

## Why Bambu Connect, not direct MQTT/FTP

Bambu Lab's local API (MQTT + FTP directly against the printer) exists and
works, but current firmware gates it behind **LAN-only Mode + Developer
Mode together** — enabling both disconnects the printer from Bambu's
cloud entirely, which means losing Bambu Handy and any remote access away
from home. If you want to keep the full Bambu ecosystem working normally,
**Bambu Connect** is Bambu Lab's own officially-sanctioned alternative: a
desktop app that stays bound to your normal Cloud-mode printer, and
accepts a sliced file from third-party software via a URL handoff. Noah
uses this path specifically so setting it up doesn't cost you Bambu Handy
or remote monitoring.

The real trade-off: Bambu Connect is a **one-directional handoff**. Noah
opens a file in Bambu Connect's window; you confirm the print *there* (a
second, separate confirmation from Noah's own chat-based one); and from
that point on, print progress shows in Bambu Connect/Handy's own UI —
Noah has no channel back to report live status, so the `PRINTER` HUD
indicator can only show whether Bambu Connect is installed, not whether
anything is actually printing.

Bambu Connect is also still in **beta** — the URL scheme this integration
relies on could change without notice.

## 1. Install Bambu Studio (for slicing)

Download and install [Bambu Studio](https://bambulab.com/en/download/studio).
Noah drives its CLI, not its GUI — confirm the installed path matches
`BAMBU_STUDIO_EXE`'s default (`C:\Program Files\Bambu Studio\bambu-studio.exe`)
or update `.env` if it's somewhere else.

## 2. Locate the P2S profile files

Bambu Studio ships bundled machine/process/filament profiles on disk, and
the CLI needs their file paths directly (`--load-settings`,
`--load-filaments`). **This step needs real verification against an actual
install** — the exact location varies by Bambu Studio version. Likely
places to check first: `%APPDATA%\BambuStudio\system\<vendor>\...` or the
install directory's `resources\profiles\` folder. Look for:
- A **machine profile** for "Bambu Lab P2S" with your nozzle size.
- A **process/quality profile** (e.g. a 0.20mm Standard preset).
- A **filament profile** (e.g. generic PLA, or whatever you print with most).

## 3. Install Bambu Connect and sign in

Download and install [Bambu Connect](https://wiki.bambulab.com/en/software/bambu-connect).
Sign into your normal Bambu account and bind it to your printer exactly
the way you already do for Bambu Handy/Studio — nothing about your
printer's own settings needs to change for this part. Bambu Connect
manages the printer connection entirely on its own; Noah only ever hands
it a file path via a URL.

## 4. Configure `.env`

Only the slicing profile paths need setting — Bambu Connect needs no
config here since it handles its own printer binding:

```
BAMBU_MACHINE_PROFILE=<full path to the P2S machine profile>
BAMBU_PROCESS_PROFILE=<full path to a process/quality profile>
BAMBU_FILAMENT_PROFILE=<full path to a filament profile>
```

## 5. Verify the handoff works

Ask Noah to generate a model ("make me a model of X"). If slicing
succeeds, it'll report an estimate and ask you to reply "print" — do that
once to confirm Noah can actually launch Bambu Connect via the
`bambu-connect://` URL scheme and that it opens with the right file
loaded, ideally on something small first.

## How Noah uses it

- `repairMesh()` (via `image23d/repair_mesh.py`) — validates/repairs the
  generated mesh before it's trusted for slicing or Fusion import.
- `sliceModel()` — invokes Bambu Studio's CLI to slice the repaired mesh.
- `isBambuConnectInstalled()` — a quick registry check (`HKCR\bambu-connect`)
  used only for the HUD's `PRINTER` indicator.
- `launchBambuConnect()` — used only after you reply "print" to a pending
  confirmation; opens `bambu-connect://import-file?path=...&name=...` so
  Bambu Connect takes over from there. Never runs automatically.

## Known limitations — read before relying on this

- **No live print status.** Once handed off, Noah can't tell you whether
  the print actually started, how far along it is, or whether it
  succeeded — that's all in Bambu Connect/Handy's own UI. The `PRINTER`
  HUD indicator only reflects whether Bambu Connect is installed.
- **Two confirmations, not one.** Noah's chat gate ("reply print") only
  hands the file to Bambu Connect — Bambu Connect then shows its own
  import/print dialog, which still needs a click there too.
- **Bambu Connect is beta software.** The `bambu-connect://import-file`
  URL scheme and its `path`/`name`/`version` parameters could change in a
  future Bambu Connect update without notice.
- **Known open issue: Bambu Studio's CLI slicer has been reported to fail
  on P2S specifically** with a `nozzle_volume_type not found` error. Noah
  detects this distinctly and tells you to slice manually in Bambu
  Studio's GUI instead if it happens.
- **Requires Bambu Connect to be installed and running** on the same
  machine as Noah's server — this isn't a remote/network handoff, it's a
  local URL-scheme launch.
