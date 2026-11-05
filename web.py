"""
Tiny built-in web server for setting up zones from a phone/laptop (no monitor needed).
Standard library only. Started by main.py; open  http://<pi-name>.local:8080

GET  /             setup page (setup.html)
GET  /stream       live MJPEG video
GET  /api/zones    current zones
POST /api/zones    replace zones  {"zones": [...]}  -> saved to zones.json, applied live
GET  /api/status   people / fps / IR mode / which lights are on
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from camera import save_zones

PAGE = Path(__file__).with_name("setup.html")
MAX_BODY = 64 * 1024


class Shared:
    """State shared between the detection loop and the web server threads."""

    def __init__(self, zones, zones_path):
        self.cond = threading.Condition()
        self.jpeg = None
        self.frame_id = 0
        self.status = {}
        self.zones = zones
        self.zones_path = zones_path
        self.zones_version = 0

    def put_frame(self, jpeg, status):
        with self.cond:
            self.jpeg = jpeg
            self.status = status
            self.frame_id += 1
            self.cond.notify_all()

    def get_zones(self):
        with self.cond:
            return self.zones, self.zones_version

    def set_zones(self, zones):
        with self.cond:
            self.zones = zones
            self.zones_version += 1
        save_zones(self.zones_path, zones)


def validate_zones(data):
    zones = data.get("zones") if isinstance(data, dict) else None
    if not isinstance(zones, list) or len(zones) > 50:
        raise ValueError("zones must be a list (max 50)")
    clean = []
    for i, z in enumerate(zones):
        pts = z.get("points")
        light = z.get("light")
        if not isinstance(light, int) or not 1 <= light <= 99:
            raise ValueError(f"zone {i}: light must be 1-99")
        if not isinstance(pts, list) or not 3 <= len(pts) <= 100:
            raise ValueError(f"zone {i}: needs 3-100 points")
        points = []
        for p in pts:
            x, y = float(p[0]), float(p[1])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"zone {i}: points must be 0..1")
            points.append([round(x, 4), round(y, 4)])
        name = str(z.get("name") or f"zone{i + 1}")[:32]
        clean.append({"name": name, "light": light, "points": points})
    return clean


class Handler(BaseHTTPRequestHandler):
    shared: Shared = None

    def log_message(self, *args):
        pass  # keep the console quiet

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/zones":
            self._send(200, {"zones": self.shared.get_zones()[0]})
        elif path == "/api/status":
            with self.shared.cond:
                self._send(200, self.shared.status)
        elif path == "/stream":
            self._stream()
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/api/zones":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise ValueError("too large")
            zones = validate_zones(json.loads(self.rfile.read(n)))
        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})
        self.shared.set_zones(zones)
        print(f"Zones updated from web: {len(zones)} zone(s)")
        self._send(200, {"ok": True, "zones": zones})

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                with self.shared.cond:
                    self.shared.cond.wait_for(lambda: self.shared.frame_id != last, timeout=2)
                    jpeg, last = self.shared.jpeg, self.shared.frame_id
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # browser closed the page


def start_server(shared, port=8080):
    handler = type("BoundHandler", (Handler,), {"shared": shared})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
