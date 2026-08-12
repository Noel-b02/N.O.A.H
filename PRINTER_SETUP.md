# Bambu Lab P2S printer setup (one-time)

Lets Noah take a generated model all the way from "make me a model of X"
through to an actual physical print: repairing/validating the mesh,
slicing it with Bambu Studio, uploading it to the printer, and starting the
print — with your explicit confirmation before anything actually prints.
This is the next stage after the image-to-3D pipeline (`HUNYUAN3D_SETUP.md`)
and Fusion 360 import (`FUSION_SETUP.md`); those two produce and preview the
model, this one gets it onto the bed.

Noah talks to the printer over the local network directly (the `mqtt` and
`basic-ftp` npm packages, no separate Python service). This needs
**Developer Mode**, not "LAN Only Mode" — those are two different toggles,
and it's worth not confusing them if you want to keep using the Bambu
ecosystem normally:

- **Developer Mode** exposes the local MQTT/FTP API this integration uses,
  *while staying connected to Bambu's cloud* — Bambu Handy, remote
  monitoring, and everything else keep working exactly as before. This is
  the one you want.
- **LAN Only Mode** is a separate, more drastic toggle that fully
  disconnects the printer from Bambu's cloud (no more remote access via
  Handy from outside your network). Noah doesn't need this — don't enable
  it unless you specifically want to drop cloud connectivity for other
  reasons.

## 1. Enable Developer Mode on the printer

On the P2S's touchscreen: Settings → find the Developer Mode toggle and
enable it (leave LAN Only Mode off). (Exact menu wording/path can shift
between firmware versions — check the printer's own screen rather than
trusting a hardcoded
path here.) This is what exposes the local MQTT/FTP services Noah connects
to; without it, the printer only talks to Bambu's cloud, which this
integration doesn't use.

## 2. Find the printer's IP, serial number, and access code

Still on the touchscreen, under the same network/LAN settings screen once
Developer Mode is on: note the printer's **IP address**, its **serial
number**, and its **LAN access code**. These three values go straight into
`.env` in step 5 — there's no default for them, they're unique to your
printer and network.

## 3. Install Bambu Studio

Download and install [Bambu Studio](https://bambulab.com/en/download/studio)
for slicing. Noah drives it via its command-line interface, not its GUI —
confirm the installed path matches `BAMBU_STUDIO_EXE`'s default
(`C:\Program Files\Bambu Studio\bambu-studio.exe`) or update `.env` if it's
somewhere else.

## 4. Locate the P2S profile files

Bambu Studio ships bundled machine/process/filament profiles on disk, and
the CLI needs their file paths directly (`--load-settings`,
`--load-filaments`). **This step needs real verification against an actual
install** — the exact location varies by Bambu Studio version. Likely
places to check first: `%APPDATA%\BambuStudio\system\<vendor>\...` or the
install directory's `resources\profiles\` folder. Look for:
- A **machine profile** for "Bambu Lab P2S" with your nozzle size.
- A **process/quality profile** (e.g. a 0.20mm Standard preset).
- A **filament profile** (e.g. generic PLA, or whatever you print with most).

Once found, set their full paths in `.env` (step 5).

## 5. Configure `.env`

```
PRINTER_IP=<printer's LAN IP>
PRINTER_SERIAL=<printer's serial number>
PRINTER_ACCESS_CODE=<printer's LAN access code>
BAMBU_MACHINE_PROFILE=<full path to the P2S machine profile>
BAMBU_PROCESS_PROFILE=<full path to a process/quality profile>
BAMBU_FILAMENT_PROFILE=<full path to a filament profile>
```

`PRINTER_ACCESS_CODE` is a real credential — your printer's local-network
password. Keep `.env` out of anything you commit or share (it's already
gitignored).

## 6. Verify MQTT connectivity

Start the server (`npm run dev`) and check the console for `[printer]
Connected.` — if it instead logs a connection error, double-check the IP,
serial, and access code, and confirm Developer Mode is still on (some
firmware updates reset it). The HUD's `PRINTER` indicator should also flip
from red `OFFLINE` to green `IDLE`.

## 7. Verify FTP + a real print

Ask Noah to generate a model ("make me a model of X"). If slicing succeeds,
it'll report an estimate and ask you to reply "print" to confirm — do that
once end-to-end to confirm the full chain (repair → slice → upload → start)
actually works on real hardware, ideally on something small/fast first.

## How Noah uses it

- `connectPrinter()` / `getPrinterStatus()` — maintains the standing MQTT
  status link, feeds the HUD's `PRINTER` indicator.
- `repairMesh()` (via `image23d/repair_mesh.py`) — validates/repairs the
  generated mesh before it's trusted for slicing or Fusion import.
- `sliceModel()` — invokes Bambu Studio's CLI to slice the repaired mesh.
- `uploadGcodeToPrinter()` / `startPrintJob()` — used only after you reply
  "print" to a pending confirmation; never runs automatically.

## Known limitations — read before relying on this

- **Bambu's local MQTT/FTP protocol is not officially documented.** This
  integration is built against reverse-engineered community knowledge, the
  same category of risk as other unofficial Bambu tooling — firmware
  updates can change behavior without notice, and the exact status-report
  field names/print-start command shape may need correcting once tested
  against your printer.
- **Known open issue: Bambu Studio's CLI slicer has been reported to fail
  on P2S specifically** with a `nozzle_volume_type not found` error. Noah
  detects this distinctly and tells you to slice manually in Bambu Studio's
  GUI instead if it happens — the rest of the pipeline (repair, connection,
  confirmation) still works either way.
- **Developer Mode must stay enabled.** If a firmware update resets it,
  Noah loses connectivity silently until you notice the HUD showing
  `PRINTER: OFFLINE` and re-check the printer's settings. This doesn't
  affect cloud/Bambu Handy access either way.
- **No cloud/remote printing** — this only works when the machine running
  Noah and the printer are on the same LAN.
- **Confirmation is required by design.** Noah will never upload or start a
  print without you replying "print" to a specific pending job first.
