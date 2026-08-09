# Fusion 360 bridge setup (one-time)

Lets Noah create geometry directly in a running Fusion 360 session. Requires
Fusion 360 already installed (which you have).

## How it works

`fusion-bridge/NoahFusionBridge/` is a Fusion 360 add-in. Once loaded, it runs
a small local HTTP server (`localhost:9000`) inside Fusion itself. When you
ask Noah to create a 3D model, `server.ts` asks the local LLM to generate
Fusion API Python code, then POSTs that code to the add-in, which executes it
against the actual open document.

## Install the add-in

1. Open Fusion 360.
2. Go to **Utilities → Add-Ins → Scripts and Add-Ins** (or press `Shift+S`).
3. On the **Add-Ins** tab, click the green **+** button next to "My Add-Ins".
4. Browse to and select this project's `fusion-bridge/NoahFusionBridge` folder.
5. Select it in the list and click **Run**. You should see a message box
   confirming it's listening on `http://localhost:9000`.
6. Optionally check **Run on Startup** so you don't have to repeat this every
   session.

## Verify it's working

With the add-in running and a Fusion document open:

```bash
curl http://localhost:9000/health
```

Should return `{"status": "ok"}`. If `curl` can't connect, the add-in isn't
running — check the Add-Ins panel for errors, or check Fusion's Text Commands
palette for the traceback.

## Try it

With both `npm run dev` and the add-in running, ask Noah something like:

> "create a 3d model of a cylinder 5cm radius and 10cm tall in fusion 360"

If it works, you'll see the geometry appear in the active Fusion document,
and Noah's reply will include the Python code that ran.

## Two different request types

Noah routes Fusion requests down one of two completely different pipelines,
based on whether the request describes parametric geometry or names a
real-world subject:

- **Parametric shapes** ("a cylinder 5cm radius", "a box 4x6x3cm", "remove
  the current models") — the local LLM writes Fusion API Python directly,
  which the bridge executes. This is the path described above.
- **Named subjects** ("a 3d model of spiderman", "make a model of a fox") —
  routed instead through the image-to-3D pipeline: Noah searches the web for
  a reference photo, runs it through a local TripoSR model to get a mesh,
  then imports that mesh into the document via the bridge's `/import-mesh`
  endpoint. Requires the separate one-time setup in `IMAGE23D_SETUP.md`.
  Quality is far rougher than a hand-modeled asset, and the mesh isn't
  real-world scaled — check `IMAGE23D_SETUP.md`'s known limitations before
  relying on it. You can ask for a target size in the same request (e.g.
  "...make it 5cm") — the mesh gets pre-scaled so its largest dimension
  matches that, before import.

The split is keyword-based (`server.ts`'s `SHAPE_WORD_PATTERN`): if the
request has no actual shape word in it (cylinder, box, sphere, etc.), it's
treated as a named subject — mentioning a size alone ("make it 5cm") doesn't
count, so it stays on the image pipeline.

**Mesh import requires a parametric document.** The bridge's `/import-mesh`
handler needs the active Fusion document in **Parametric** design mode (the
default) — a document switched to **Direct Modeling** mode has no timeline,
and Fusion's mesh-import API behaves differently there. The add-in detects
which mode the document is in and adjusts automatically, but if you've
deliberately switched a document to direct modeling, mesh import will use a
different (also supported) code path than parametric-shape creation does.

## Known limitations — read before relying on this

- **The local model doesn't know Fusion's API deeply.** It'll generally
  handle simple parametric shapes (boxes, cylinders, basic extrusions), but
  more complex requests (assemblies, fillets, patterns, sketches with
  constraints) are much more likely to fail or produce something wrong.
  When it fails, Noah's reply includes the actual Fusion error traceback and
  the code that was attempted — that's meant to help you (or a follow-up
  conversation) refine the prompt in `server.ts`'s `looksLikeFusionRequest`
  block, not as a dead end.
- **Test in a scratch document first**, not a real project file, until
  you've got a feel for how reliable this is on your machine — the generated
  code runs directly in Fusion with an `exec()` call.
- **The trigger phrase is deliberately narrow** — it requires wording like
  "3d model", "cad model", or "in fusion 360" alongside a creation verb
  (see `FUSION_TRIGGER_PATTERN` in `server.ts`), specifically so this doesn't
  accidentally fire and execute code during an ordinary conversation.
