"""Part 1: detect, trace and locate the centres of shapes in a single image.

Usage:
  python detect_image.py media/static.png -o results/static_detected.png
  python detect_image.py media/static.png -o results/static_3d.png --3d
"""

import argparse
from pathlib import Path

import cv2

from pennair.camera import PlanarLocalizer
from pennair.detector import ShapeDetector
from pennair.draw import draw_detections


def main():
    p = argparse.ArgumentParser(description="Detect shapes in one image.")
    p.add_argument("image")
    p.add_argument("-o", "--output", default="results/static_detected.png")
    p.add_argument("--3d", dest="three_d", action="store_true", help="also estimate X, Y, Z (Part 4)")
    p.add_argument("--principal-point", choices=["K", "center"], default="K",
                   help="use cx, cy from the given K (0, 0), or the image centre")
    args = p.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"could not read {args.image}")

    detections = ShapeDetector().detect(frame)
    positions = None
    if args.three_d:
        localizer = PlanarLocalizer(principal_point=args.principal_point)
        positions = localizer.localize(detections, frame.shape[1], frame.shape[0])

    for i, d in enumerate(detections):
        line = f"{d.label:10s} centre=({d.center[0]:7.1f}, {d.center[1]:7.1f}) px  area={d.area:8.0f} px"
        if positions is not None and positions[i] is not None:
            line += "  X={:7.1f} Y={:7.1f} Z={:6.1f} in".format(*positions[i])
        print(line)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, draw_detections(frame, detections, positions))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
