"""Score background agnosticism using the easy video as ground truth.

Both challenge videos move the same five shapes along the same paths; only the
background and the shape fills differ. So for the same frame index, detections
on the hard video should match detections on the easy one. For each frame this
prints, per shape: centre error (px), area ratio hard/easy, and both labels.

Usage: python tools/compare_backgrounds.py 0 460 920 1380 1836
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pennair.detector import ShapeDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main(indices):
    det = ShapeDetector()
    worst_err, worst_ratio = 0.0, 1.0
    for i in indices:
        easy = det.detect(read_frame(ROOT / "media/dynamic.mp4", i))
        hard = det.detect(read_frame(ROOT / "media/dynamic_hard.mp4", i))
        print(f"frame {i}: easy={len(easy)} hard={len(hard)}")
        for e in easy:
            if not hard:
                break
            h = min(hard, key=lambda d: np.hypot(d.center[0] - e.center[0], d.center[1] - e.center[1]))
            err = float(np.hypot(h.center[0] - e.center[0], h.center[1] - e.center[1]))
            ratio = h.area / e.area
            worst_err = max(worst_err, err)
            if abs(ratio - 1) > abs(worst_ratio - 1):
                worst_ratio = ratio
            flag = "" if err < 5 and abs(ratio - 1) < 0.05 and h.label == e.label else "   <--"
            print(f"  {e.label:10s} -> {h.label:10s} centre err {err:6.1f}px  area ratio {ratio:5.3f}{flag}")
    print(f"worst centre error {worst_err:.1f}px, worst area ratio {worst_ratio:.3f}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0, 460, 920, 1380, 1836])
