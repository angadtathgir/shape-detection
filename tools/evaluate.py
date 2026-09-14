"""Measure accuracy against ground truth recovered from the easy video.

In the easy video every shape is one flat colour, so its true mask can be
recovered almost exactly: pixels within a small tolerance of the shape's colour,
near the shape, largest connected piece, holes filled. Both videos move
identical shapes along identical paths, so the same masks are ground truth for
the hard video too.

For every true shape this reports the IoU (intersection over union) of the best
overlapping detection, and the distance between their centres.

Usage: python tools/evaluate.py [frame indices...] [--save-gt path.png]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pennair.detector import ShapeDetector  # noqa: E402


def read_frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    frames = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames[i] = frame
    cap.release()
    return frames


def filled_mask(contour, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, [contour], -1, 1, -1)
    return m


def ground_truth(frame, tol=10):
    """True masks for the flat-coloured shapes in an easy-video frame.

    A reference detection only tells us roughly where each shape is and what
    colour it is; the mask itself comes purely from colour matching.
    """
    masks = []
    for d in ShapeDetector().detect(frame):
        region = filled_mask(d.contour, frame.shape)
        inner = cv2.erode(region, np.ones((15, 15), np.uint8))
        if inner.sum() < 100:
            continue
        colour = np.median(frame[inner > 0], axis=0)
        near = cv2.dilate(region, np.ones((31, 31), np.uint8))
        match = (np.abs(frame.astype(np.int16) - colour).max(axis=2) <= tol) & (near > 0)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(match.astype(np.uint8), connectivity=4)
        if n < 2:
            continue
        biggest = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        contours, _ = cv2.findContours(biggest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        masks.append(filled_mask(max(contours, key=cv2.contourArea), frame.shape))
    return masks


def score(detections, truths, shape):
    """For each true mask: (IoU, centre error px, label) of the best detection."""
    det_masks = [filled_mask(d.contour, shape) for d in detections]
    rows = []
    for truth in truths:
        ys, xs = np.nonzero(truth)
        tc = (xs.mean(), ys.mean())
        best = (0.0, float("nan"), "missed")
        for d, m in zip(detections, det_masks):
            inter = np.count_nonzero(m & truth)
            if inter:
                iou = inter / np.count_nonzero(m | truth)
                if iou > best[0]:
                    best = (iou, float(np.hypot(d.center[0] - tc[0], d.center[1] - tc[1])), d.label)
        rows.append(best)
    return rows


def evaluate(detector, indices, verbose=False):
    easy = read_frames(ROOT / "media/dynamic.mp4", indices)
    hard = read_frames(ROOT / "media/dynamic_hard.mp4", indices)
    results = {"easy": [], "hard": []}
    for i in indices:
        truths = ground_truth(easy[i])
        for name, frames in (("easy", easy), ("hard", hard)):
            rows = score(detector.detect(frames[i]), truths, frames[i].shape)
            results[name] += rows
            if verbose:
                for iou, err, label in rows:
                    if iou < 0.95:
                        print(f"  frame {i:4d} {name}: IoU {iou:.3f}, centre err {err:.1f}px ({label})")
    return results


def summarise(results):
    for name, rows in results.items():
        iou = np.array([r[0] for r in rows])
        err = np.array([r[1] for r in rows if not np.isnan(r[1])])
        print(f"{name}: {len(rows)} shapes, IoU mean {iou.mean():.4f} min {iou.min():.3f}, "
              f"centre err mean {err.mean():.2f}px max {err.max():.2f}px")


if __name__ == "__main__":
    args = sys.argv[1:]
    save = None
    if "--save-gt" in args:
        k = args.index("--save-gt")
        save = args[k + 1]
        del args[k:k + 2]
    indices = [int(a) for a in args] or list(range(0, 1837, 60))
    if save:
        frame = read_frames(ROOT / "media/dynamic.mp4", [indices[0]])[indices[0]]
        vis = frame.copy()
        for m in ground_truth(frame):
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, contours, -1, (255, 0, 255), 2)
        cv2.imwrite(save, vis)
    summarise(evaluate(ShapeDetector(), indices, verbose=True))
