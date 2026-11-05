"""Shared camera + IR helpers used by main.py and zone_editor.py."""

import json
import time

import cv2
import numpy as np


class Camera:
    """Picamera2 (CSI IR camera) with an OpenCV fallback (USB camera)."""

    def __init__(self, source="picam", width=768, height=432):
        self.picam = None
        self.cap = None
        if source == "picam":
            from picamera2 import Picamera2

            self.picam = Picamera2()
            main = {"size": (width, height), "format": "RGB888"}  # RGB888 = BGR byte order
            try:
                # IMX708 (Camera Module 3): 2304x1296 is the binned mode that uses the WHOLE
                # sensor, i.e. the full 120 deg field of view. Smaller modes crop the middle.
                cfg = self.picam.create_video_configuration(
                    main=main, sensor={"output_size": (2304, 1296)})
            except Exception:
                cfg = self.picam.create_video_configuration(main=main)
            self.picam.configure(cfg)
            self.picam.start()
            try:  # make sure no digital zoom/crop is applied: use the full sensor area
                self.picam.set_controls({"ScalerCrop": self.picam.camera_controls["ScalerCrop"][1]})
            except Exception:
                pass
            try:  # Camera Module 3 has autofocus; keep it refocusing
                from libcamera import controls
                self.picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            except Exception:
                pass
            time.sleep(1.0)  # let auto-exposure settle
        else:
            src = int(source) if source.isdigit() else source
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera source {source!r}")

    def read(self):
        if self.picam is not None:
            frame = self.picam.capture_array()  # BGR numpy array
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None
        if frame.ndim == 3 and frame.shape[2] == 4:  # XBGR -> BGR, just in case
            frame = frame[:, :, :3]
        return frame

    def close(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


def is_ir_lit(frame, threshold=25):
    """IR-lit scenes have almost no colour (everything is grey/purple-ish),
    daylight scenes do. Low average saturation -> camera is seeing mostly IR."""
    small = cv2.resize(frame, (80, 60))
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    return sat < threshold, sat


def ir_preprocess(frame):
    """NoIR cameras give a pink/purple cast. Grayscale + contrast boost, back to 3 channels,
    gives YOLO a consistent image day or night."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------- zones ----------
# zones.json format (points are 0..1 fractions of the frame, so resolution doesn't matter):
# {"zones": [{"name": "door", "light": 1, "points": [[0.1, 0.5], [0.4, 0.5], [0.4, 0.9]]}, ...]}

ZONE_COLORS = [(0, 200, 255), (255, 120, 0), (0, 255, 120), (255, 0, 200),
               (120, 255, 255), (255, 255, 0), (180, 120, 255), (0, 120, 255), (200, 200, 200)]


def load_zones(path):
    try:
        with open(path) as f:
            return json.load(f).get("zones", [])
    except FileNotFoundError:
        return []


def save_zones(path, zones):
    with open(path, "w") as f:
        json.dump({"zones": zones}, f, indent=2)


def zone_pixels(zone, w, h):
    return np.array([[int(x * w), int(y * h)] for x, y in zone["points"]], dtype=np.int32)


def draw_zones(img, zones, active_lights=()):
    """Outline every zone; fill the ones whose light is currently on."""
    h, w = img.shape[:2]
    overlay = img.copy()
    for z in zones:
        pts = zone_pixels(z, w, h)
        color = ZONE_COLORS[(z["light"] - 1) % len(ZONE_COLORS)]
        if z["light"] in active_lights:
            cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        x, y = pts[0]
        cv2.putText(img, f'{z["name"]} -> L{z["light"]}', (int(x) + 4, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, dst=img)
    return img
