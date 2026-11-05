"""
YOLO person detection on a Raspberry Pi with an IR (NoIR) camera, with per-zone lights.

Files (copy all to the Pi):  main.py  camera.py  web.py  setup.html  (zone_editor.py optional)

Run:   python3 main.py                          headless; setup page at http://<pi>.local:8080
       python3 main.py --serial /dev/ttyUSB0    also drive the ESP32
       DISPLAY=:0 python3 main.py --show        also show a window on the Pi's monitor

Zones are drawn on the setup page from any phone/laptop on the same Wi-Fi and apply
immediately (saved to zones.json). A person is "in" a zone when their FEET (bottom-centre
of their box) are inside it; use --point center to use the middle of the box instead.

Serial messages to the ESP32 (one per line):  "ON 1", "OFF 1", "ON 2", ...
With no zones, the whole frame is one zone driving light 1.

Faster on Pi: yolo export model=yolo11n.pt format=ncnn imgsz=320
              then add  --model yolo11n_ncnn_model
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from camera import Camera, draw_zones, ir_preprocess, is_ir_lit, load_zones, zone_pixels
import web

PERSON_CLASS = 0  # COCO class id for "person"
HERE = Path(__file__).resolve().parent
WHOLE_FRAME = [{"name": "all", "light": 1, "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO person detection (IR camera) with zones")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights or exported model dir")
    p.add_argument("--imgsz", type=int, default=416, help="inference size (bigger finds people farther away, but slower)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--source", default="picam", help="'picam' or an OpenCV index/path, e.g. 0")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--zones", default=str(HERE / "zones.json"), help="zone file")
    p.add_argument("--point", choices=["feet", "center"], default="feet",
                   help="which point of a person decides their zone")
    p.add_argument("--port", type=int, default=8080, help="setup web page port (0 = off)")
    p.add_argument("--show", action="store_true", help="show annotated preview window")
    p.add_argument("--serial", default=None, help="serial port of ESP32, e.g. /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--on-frames", type=int, default=3, help="consecutive hits before ON")
    p.add_argument("--off-seconds", type=float, default=5.0, help="zone empty this long -> OFF")
    p.add_argument("--no-gray", action="store_true", help="don't convert IR frames to grayscale")
    return p.parse_args()


def open_serial(port, baud):
    if not port:
        return None
    import serial

    s = serial.Serial(port, baud, timeout=0)
    time.sleep(2.0)  # ESP32 resets when the port opens
    return s


class Light:
    """Debounced on/off state for one light number."""

    def __init__(self, num):
        self.num = num
        self.on = False
        self.streak = 0
        self.last_seen = 0.0


def main():
    args = parse_args()

    shared = web.Shared(load_zones(args.zones), args.zones)
    if args.port:
        web.start_server(shared, args.port)
        print(f"Setup page: http://<this-pi>.local:{args.port}")

    model = YOLO(args.model)
    cam = Camera(args.source, args.width, args.height)
    ser = open_serial(args.serial, args.baud)

    lights = {}        # light number -> Light
    zones = []         # zones currently in use
    zone_px = []       # (light, polygon in pixels)
    zones_version = -1
    fps_t, fps_n, fps = time.time(), 0, 0.0

    def set_light(light, on):
        if on == light.on:
            return
        light.on = on
        print(f"[{time.strftime('%H:%M:%S')}] light {light.num} {'ON' if on else 'OFF'}")
        if ser:
            ser.write(f"{'ON' if on else 'OFF'} {light.num}\n".encode())

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Frame grab failed")
                time.sleep(0.1)
                continue
            h, w = frame.shape[:2]

            # pick up zone changes from the web page
            user_zones, version = shared.get_zones()
            if version != zones_version:
                zones_version = version
                zones = user_zones or WHOLE_FRAME
                zone_px = [(z["light"], zone_pixels(z, w, h)) for z in zones]
                wanted = {z["light"] for z in zones}
                for num in list(lights):
                    if num not in wanted:  # light no longer used by any zone
                        set_light(lights.pop(num), False)
                for num in wanted:
                    lights.setdefault(num, Light(num))
                print(f"Using {len(user_zones)} zone(s), lights {sorted(wanted)}")

            ir_mode, sat = is_ir_lit(frame)
            img = frame if args.no_gray else ir_preprocess(frame)

            res = model.predict(
                img, imgsz=args.imgsz, conf=args.conf, classes=[PERSON_CLASS], verbose=False
            )[0]

            # which lights have a person in one of their zones this frame
            points = []
            for x1, y1, x2, y2 in res.boxes.xyxy.tolist():
                px = (x1 + x2) / 2
                py = y2 if args.point == "feet" else (y1 + y2) / 2
                points.append((float(px), float(min(py, h - 1))))
            occupied = {
                num for num, poly in zone_px
                if any(cv2.pointPolygonTest(poly, pt, False) >= 0 for pt in points)
            }

            now = time.time()
            for num, light in lights.items():
                if num in occupied:
                    light.streak += 1
                    light.last_seen = now
                    if light.streak >= args.on_frames:
                        set_light(light, True)
                else:
                    light.streak = 0
                    if light.on and now - light.last_seen > args.off_seconds:
                        set_light(light, False)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            lights_on = sorted(n for n, l in lights.items() if l.on)

            if args.port or args.show:
                out = res.plot()  # person boxes
                for pt in points:
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)

            if args.port:  # web page draws zones itself, so stream without them
                ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    shared.put_frame(jpg.tobytes(), {
                        "people": len(points), "fps": round(fps, 1), "lights_on": lights_on,
                        "mode": "IR" if ir_mode else "DAY", "sat": round(float(sat), 1),
                    })

            if args.show:
                draw_zones(out, zones, set(lights_on))
                cv2.putText(out, f"people: {len(points)}  fps: {fps:.1f}  "
                                 f"lights on: {','.join(map(str, lights_on)) or '-'}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(out, f"{'IR' if ir_mode else 'DAY'} (sat {sat:.0f})", (8, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if ir_mode else (0, 255, 255), 2)
                cv2.imshow("person detection", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for light in lights.values():
            set_light(light, False)
        cam.close()
        if ser:
            ser.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
