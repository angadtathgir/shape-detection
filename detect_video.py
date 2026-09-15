"""Parts 2-4: run the detector on a video as a stream, one frame at a time.

The loop only ever sees the current frame (plus the last known depth in 3D
mode), the same way the aircraft receives its camera feed.

Usage:
  python detect_video.py media/dynamic.mp4 -o results/dynamic_detected.mp4
  python detect_video.py media/dynamic_hard.mp4 -o results/dynamic_hard_detected.mp4
  python detect_video.py media/dynamic_hard.mp4 -o results/dynamic_hard_3d.mp4 --3d --csv results/dynamic_hard_3d.csv
"""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pennair.camera import PlanarLocalizer
from pennair.detector import ShapeDetector
from pennair.draw import draw_detections


def open_writer(path, fps, size):
    # H.264 ("avc1") plays in browsers; "mp4v" is the fallback if it's unavailable.
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, fourcc
    raise SystemExit(f"could not open a video writer for {path}")


def main():
    p = argparse.ArgumentParser(description="Detect shapes in a video, frame by frame.")
    p.add_argument("video")
    p.add_argument("-o", "--output", default="results/detected.mp4")
    p.add_argument("--3d", dest="three_d", action="store_true", help="also estimate X, Y, Z (Part 4)")
    p.add_argument("--principal-point", choices=["K", "center"], default="K",
                   help="use cx, cy from the given K (0, 0), or the image centre")
    p.add_argument("--out-width", type=int, default=960, help="width of the saved video")
    p.add_argument("--csv", help="optional per-detection log")
    p.add_argument("--show", action="store_true", help="display while processing (q to quit)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (args.out_width, round(h * args.out_width / w))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer, fourcc = open_writer(args.output, fps, out_size)
    csv_file = log = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        log = csv.writer(csv_file)
        log.writerow(["frame", "label", "u_px", "v_px", "area_px", "X_in", "Y_in", "Z_in"])

    detector = ShapeDetector()
    localizer = PlanarLocalizer(principal_point=args.principal_point) if args.three_d else None
    times, counts, depths = [], Counter(), []

    index = 0
    while True:
        ok, frame = cap.read()  # one frame at a time: no look-ahead
        if not ok:
            break
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        positions = localizer.localize(detections, w, h) if localizer else None
        times.append(time.perf_counter() - t0)
        counts[len(detections)] += 1
        # Depth statistics only count frames where the circle is actually visible
        # (on other frames the last accepted depth is just carried forward).
        if localizer and localizer.depth is not None and any(d.label == "circle" for d in detections):
            depths.append(localizer.depth)

        if log:
            for i, d in enumerate(detections):
                xyz = positions[i] if positions and positions[i] else ("", "", "")
                log.writerow([index, d.label, f"{d.center[0]:.1f}", f"{d.center[1]:.1f}",
                              f"{d.area:.0f}", *[f"{c:.2f}" if c != "" else "" for c in xyz]])

        vis = cv2.resize(draw_detections(frame, detections, positions), out_size,
                         interpolation=cv2.INTER_AREA)
        writer.write(vis)
        if args.show:
            cv2.imshow("PennAiR shapes", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        index += 1

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()

    ms = np.array(times) * 1000
    print(f"{index} frames from {args.video} -> {args.output} ({fourcc})")
    print(f"detection time: mean {ms.mean():.1f} ms, p95 {np.percentile(ms, 95):.1f} ms "
          f"(~{1000 / ms.mean():.0f} fps; video is {fps:.0f} fps)")
    print("shapes found per frame:", dict(sorted(counts.items())))
    if depths:
        d = np.array(depths)
        print(f"circle depth Z over {len(d)} frames with the circle visible: median {np.median(d):.2f} in, "
              f"std {d.std():.2f} in, range {d.min():.2f}-{d.max():.2f} in")


if __name__ == "__main__":
    main()
