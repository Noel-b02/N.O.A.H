import adsk.core
import adsk.fusion
import math
import traceback
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# Fusion's API is not thread-safe: geometry can only be created/modified from
# the main UI thread. The HTTP server below runs on a background thread (to
# avoid blocking Fusion while waiting for requests), so incoming code hands
# off to the main thread via Fusion's custom-event mechanism — the standard,
# documented pattern for external control of a running Fusion session — and
# the HTTP thread blocks on a threading.Event until that handler signals it's
# done, so the HTTP response carries the real success/failure result.

app = None
ui = None
handlers = []
httpServer = None
httpServerThread = None

CUSTOM_EVENT_ID = "NoahFusionBridgeExecute"
IMPORT_MESH_EVENT_ID = "NoahFusionBridgeImportMesh"
PORT = 9000

pendingCode = None
pendingMeshPath = None
pendingResult = {}
resultReady = threading.Event()


class ExecuteCustomEventHandler(adsk.core.CustomEventHandler):
    def notify(self, args):
        global pendingResult
        try:
            design = adsk.fusion.Design.cast(app.activeProduct)
            if design is None:
                pendingResult = {
                    "success": False,
                    "error": "No active Fusion Design document — open or create one first."
                }
                return

            # Variables available to generated code when it runs. math is
            # included directly (rather than relying on generated code to
            # "import math" itself) since that's proven unreliable in
            # practice — generated code has used math.pi without importing
            # it, which would otherwise fail with NameError.
            execGlobals = {
                "adsk": adsk,
                "app": app,
                "ui": ui,
                "design": design,
                "rootComp": design.rootComponent,
                "math": math
            }
            exec(pendingCode, execGlobals)

            # Geometry created via a custom-event handler (rather than
            # through Fusion's normal UI command flow) doesn't always trigger
            # an automatic viewport redraw — the document model is updated
            # correctly, but the 3D view can just keep showing its old state.
            # fit() additionally re-frames the camera so new geometry isn't
            # just correctly drawn but actually in view.
            app.activeViewport.refresh()
            app.activeViewport.fit()

            pendingResult = {"success": True, "error": None}
        except Exception:
            pendingResult = {"success": False, "error": traceback.format_exc()}
        finally:
            resultReady.set()


class ImportMeshCustomEventHandler(adsk.core.CustomEventHandler):
    def notify(self, args):
        global pendingResult
        try:
            design = adsk.fusion.Design.cast(app.activeProduct)
            if design is None:
                pendingResult = {
                    "success": False,
                    "error": "No active Fusion Design document — open or create one first."
                }
                return

            rootComp = design.rootComponent

            # Base features only exist in parametric documents — in a
            # direct-modeling document (no timeline) BaseFeatures.add()
            # itself throws, and meshBodies.add() can be called directly
            # with no base/form feature at all since there's no edit
            # session to manage.
            if design.designType == adsk.fusion.DesignTypes.ParametricDesignType:
                baseFeature = rootComp.features.baseFeatures.add()
                baseFeature.startEdit()
                rootComp.meshBodies.add(pendingMeshPath, adsk.fusion.MeshUnits.CentimeterMeshUnit, baseFeature)
                baseFeature.finishEdit()
            else:
                rootComp.meshBodies.add(pendingMeshPath, adsk.fusion.MeshUnits.CentimeterMeshUnit)

            app.activeViewport.refresh()
            app.activeViewport.fit()

            pendingResult = {"success": True, "error": None}
        except Exception:
            pendingResult = {"success": False, "error": traceback.format_exc()}
        finally:
            resultReady.set()


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        global pendingCode, pendingMeshPath, pendingResult

        if self.path == "/execute":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                code = data.get("code", "")
            except Exception as e:
                self._send_json(400, {"success": False, "error": f"Invalid request: {e}"})
                return

            if not code.strip():
                self._send_json(400, {"success": False, "error": "code is required"})
                return

            pendingCode = code
            pendingResult = {}
            resultReady.clear()

            app.fireCustomEvent(CUSTOM_EVENT_ID)

            completed = resultReady.wait(timeout=30)
            if not completed:
                self._send_json(200, {"success": False, "error": "Execution timed out after 30s"})
            else:
                self._send_json(200, pendingResult)

        elif self.path == "/import-mesh":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                meshPath = data.get("path", "")
            except Exception as e:
                self._send_json(400, {"success": False, "error": f"Invalid request: {e}"})
                return

            if not meshPath.strip():
                self._send_json(400, {"success": False, "error": "path is required"})
                return

            pendingMeshPath = meshPath
            pendingResult = {}
            resultReady.clear()

            app.fireCustomEvent(IMPORT_MESH_EVENT_ID)

            # Mesh import can take longer than a small parametric exec() on
            # larger files, so this gets a longer wait than /execute's.
            completed = resultReady.wait(timeout=60)
            if not completed:
                self._send_json(200, {"success": False, "error": "Mesh import timed out after 60s"})
            else:
                self._send_json(200, pendingResult)

        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass  # suppress default request logging into Fusion's Text Commands console


def run(context):
    global app, ui, httpServer, httpServerThread

    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        customEvent = app.registerCustomEvent(CUSTOM_EVENT_ID)
        onExecute = ExecuteCustomEventHandler()
        customEvent.add(onExecute)
        handlers.append(onExecute)

        importMeshEvent = app.registerCustomEvent(IMPORT_MESH_EVENT_ID)
        onImportMesh = ImportMeshCustomEventHandler()
        importMeshEvent.add(onImportMesh)
        handlers.append(onImportMesh)

        httpServer = HTTPServer(("localhost", PORT), RequestHandler)
        httpServerThread = threading.Thread(target=httpServer.serve_forever, daemon=True)
        httpServerThread.start()

        ui.messageBox(f"N.O.A.H Fusion bridge is listening on http://localhost:{PORT}")
    except Exception:
        if ui:
            ui.messageBox("Failed to start N.O.A.H Fusion bridge:\n{}".format(traceback.format_exc()))


def stop(context):
    global httpServer

    try:
        if httpServer:
            httpServer.shutdown()
            httpServer.server_close()
            httpServer = None

        app.unregisterCustomEvent(CUSTOM_EVENT_ID)
        app.unregisterCustomEvent(IMPORT_MESH_EVENT_ID)
    except Exception:
        if ui:
            ui.messageBox("Failed to stop N.O.A.H Fusion bridge cleanly:\n{}".format(traceback.format_exc()))
