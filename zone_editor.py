"""
Draw zones on the live camera picture and assign each zone a light number.
Needs the Pi's monitor:   DISPLAY=:0 python3 zone_editor.py

Mouse / keys (click the window first so it gets key presses):
    left click      add a corner to the zone you're drawing
    right click     finish the zone (or press Enter)  - needs 3+ corners
    1-9             light number for the NEXT zone you finish
    u               undo last corner
    d               delete last finished zone
    c               clear all zones
    s               save to zones.json
    q / Esc         quit (asks nothing - press s first!)
"""

import argparse

import cv2

from camera import Camera, ZONE_COLORS, draw_zones, load_zones, save_zones

WIN = "zone editor"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zones", default="zones.json")
    p.add_argument("--source", default="picam")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    args = p.parse_args()

    cam = Camera(args.source, args.width, args.height)
    zones = load_zones(args.zones)
    current = []          # corners (pixels) of the zone being drawn
    state = {"light": 1, "msg": f"loaded {len(zones)} zone(s)" if zones else "no zones yet"}

    frame = cam.read()
    h, w = frame.shape[:2]

    def finish_zone():
        if len(current) < 3:
            state["msg"] = "need at least 3 corners"
            return
        n = len(zones) + 1
        zones.append({
            "name": f"zone{n}",
            "light": state["light"],
            "points": [[round(x / w, 4), round(y / h, 4)] for x, y in current],
        })
        current.clear()
        state["msg"] = f"added zone{n} -> light {state['light']}"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            finish_zone()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    try:
        while True:
            f = cam.read()
            if f is not None:
                frame = f
            img = draw_zones(frame.copy(), zones)

            color = ZONE_COLORS[(state["light"] - 1) % len(ZONE_COLORS)]
            for i, pt in enumerate(current):
                cv2.circle(img, pt, 4, color, -1)
                if i:
                    cv2.line(img, current[i - 1], pt, color, 2)
            if len(current) > 2:
                cv2.line(img, current[-1], current[0], color, 1)  # preview closing edge

            cv2.rectangle(img, (0, h - 44), (w, h), (0, 0, 0), -1)
            cv2.putText(img, f"next zone -> light {state['light']}   |   {state['msg']}",
                        (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, "L-click corner  R-click/Enter finish  1-9 light  u undo  "
                             "d del  c clear  s save  q quit",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
            cv2.imshow(WIN, img)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k in (13, 10):
                finish_zone()
            elif ord("1") <= k <= ord("9"):
                state["light"] = k - ord("0")
                state["msg"] = f"light set to {state['light']}"
            elif k == ord("u") and current:
                current.pop()
            elif k == ord("d") and zones:
                state["msg"] = f"deleted {zones.pop()['name']}"
            elif k == ord("c"):
                zones.clear()
                current.clear()
                state["msg"] = "cleared"
            elif k == ord("s"):
                save_zones(args.zones, zones)
                state["msg"] = f"saved {len(zones)} zone(s) to {args.zones}"
                print(state["msg"])
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
