"""Background-agnostic shape detection (Parts 1-3).

Idea: the backgrounds (grass, gravel) are rough at small scales, while every
shape, even a gradient-filled one, is smooth. So instead of asking "what colour
is the background?", we ask "where is the image locally smooth?".

Pipeline per frame:
  1. Downscale to a fixed processing width (speed).
  2. Texture map: subtract a lightly blurred copy of the frame, which removes
     smooth shading such as colour gradients, then average what is left over a
     small window. Done per Lab channel, keeping the max, so an edge in colour
     alone still counts.
  3. Seeds: pixels far below the frame's median texture. The threshold is
     relative, so it adapts to each background. Seeds then grow through
     "fairly smooth" pixels connected to them (hysteresis, as in Canny).
  4. Watershed: seeds flood a narrow band around themselves and meet the
     background at the strongest colour edge, snapping outlines onto the true
     boundary (the texture window shrinks seeds by a few pixels at every edge).
  5. Per region: outline traced at the input resolution, centroid from image
     moments, and a shape label from the outline's corners.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    contour: np.ndarray  # (N, 1, 2) int32, in input-frame pixel coordinates
    center: tuple        # (u, v) float, centroid in input-frame pixels
    area: float          # region area in input-frame pixels
    label: str           # circle / triangle / rectangle / trapezoid / pentagon / partial / ...


class ShapeDetector:
    def __init__(self, proc_width=960, sigma=1.5, window=7, seed_ratio=0.3,
                 grow_ratio=0.5, min_area_frac=1e-3, band=8, edge_blur=7):
        self.proc_width = proc_width        # frames wider than this are downscaled
        self.sigma = sigma                  # blur used to strip smooth shading (px)
        self.window = window                # averaging window for the texture map (px, odd)
        self.seed_ratio = seed_ratio        # strict: seed if texture < ratio * median texture
        self.grow_ratio = grow_ratio        # loose: seeds may grow through texture < ratio * median
        self.min_area_frac = min_area_frac  # ignore regions smaller than this fraction of the frame
        self.band = band                    # width of the band the watershed decides (px)
        self.edge_blur = edge_blur          # median filter before the watershed (0 = off)
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))

    # ---- step 2 -------------------------------------------------------------
    def texture_map(self, bgr):
        """High-frequency energy: mean |I - blur(I)| over a window, max over L, a, b.

        Blurring leaves anything that varies slowly almost unchanged (a linear
        ramp is exactly unchanged), so a shape's colour gradient leaves a
        residual near zero while grass and gravel texture survives.

        Plain local standard deviation was tried first and failed on the hard
        video: a clean brightness ramp scored as rough as gravel, and the white
        trapezoid lost half its area.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        residual = cv2.absdiff(lab, cv2.GaussianBlur(lab, (0, 0), self.sigma))
        energy = cv2.boxFilter(residual, -1, (self.window, self.window))
        return energy.max(axis=2)

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, self.proc_width / w)
        small = frame if scale == 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # ---- step 3: seeds, then hysteresis growth ------------------------------
        texture = self.texture_map(small)
        # Relative thresholds: grass and gravel have very different absolute
        # texture, but in both the shapes sit far below the median.
        med = float(np.median(texture))
        seed_t = max(0.5, self.seed_ratio * med)
        grow_t = max(seed_t, self.grow_ratio * med)
        strict = (texture < seed_t).astype(np.uint8)
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, self._open)  # drop speckle

        n, labels, stats, _ = cv2.connectedComponentsWithStats(strict, connectivity=4)
        min_area = self.min_area_frac * small.shape[0] * small.shape[1]
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if not keep:
            return []
        lut = np.zeros(n, np.int32)
        lut[keep] = np.arange(2, len(keep) + 2)
        markers = lut[labels]                    # 2.. = one id per seed, 0 elsewhere
        seeds = (markers > 1).astype(np.uint8)

        # Keep "fairly smooth" pixels only if they connect to a seed, so a seed
        # absorbs slightly rough parts of its own shape without picking up
        # smooth patches of background.
        loose = ((texture < grow_t) | (seeds > 0)).astype(np.uint8)
        n2, labels2 = cv2.connectedComponents(loose, connectivity=4)
        has_seed = np.zeros(n2, bool)
        has_seed[np.unique(labels2[seeds > 0])] = True
        grown = has_seed[labels2].astype(np.uint8)

        # ---- step 4: watershed refinement -------------------------------------
        # Marker 1 = background (more than `band` px outside the grown regions),
        # 2.. = seeds, 0 = unknown. The watershed assigns the unknown pixels, so
        # outlines snap to real colour edges; the narrow band caps how far an
        # outline can leak where the edge is weak (green shape on green grass).
        near = cv2.dilate(grown, self._band)
        markers[near == 0] = 1
        # A median filter flattens thin texture (grass blades) but keeps straight
        # step edges in place, so the background floods less eagerly across a
        # weak shape edge.
        ws_image = cv2.medianBlur(small, self.edge_blur) if self.edge_blur else small
        cv2.watershed(ws_image, markers)  # boundaries come back as -1
        watershed_line = markers == -1

        # ---- step 5: per-region measurements ------------------------------------
        detections = []
        one = np.ones((3, 3), np.uint8)
        for region_id in range(2, len(keep) + 2):
            core = (markers == region_id).astype(np.uint8)
            # The 1-px watershed line sits on the edge itself: count the part of
            # it touching this region as half inside, half outside.
            line = (cv2.dilate(core, one) > 0) & watershed_line
            soft = core.astype(np.float32) + 0.5 * line
            contours, _ = cv2.findContours((soft > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if x <= 1 or y <= 1 or x + bw >= small.shape[1] - 1 or y + bh >= small.shape[0] - 1:
                label = "partial"  # cut off by the frame edge: not a whole shape
            else:
                label = classify(cnt)
            outline = trace_at_input_resolution(soft, cnt, frame.shape, small.shape)
            # Centre = centroid of the region (first image moments / area).
            m = cv2.moments(outline)
            if m["m00"] == 0:
                continue
            detections.append(Detection(
                contour=outline,
                center=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
                area=m["m00"],
                label=label,
            ))
        return detections


def trace_at_input_resolution(soft, cnt, full_shape, small_shape):
    """Trace a region's outline on its mask upsampled to the input frame size.

    Multiplying a half-resolution contour by 2 snaps every point to an even
    pixel and shifts it by half a pixel (~2 px of boundary error at 1080p).
    Bilinear upsampling of the region's bounding box, thresholded at 0.5, keeps
    the edge's sub-pixel position.
    """
    sx = full_shape[1] / small_shape[1]
    sy = full_shape[0] / small_shape[0]
    x, y, bw, bh = cv2.boundingRect(cnt)
    x0, y0 = max(x - 2, 0), max(y - 2, 0)
    x1, y1 = min(x + bw + 2, small_shape[1]), min(y + bh + 2, small_shape[0])
    roi = soft[y0:y1, x0:x1]
    if sx != 1.0 or sy != 1.0:
        roi = cv2.resize(roi, (round((x1 - x0) * sx), round((y1 - y0) * sy)),
                         interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours((roi >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.round(cnt * [sx, sy]).astype(np.int32)
    best = max(contours, key=cv2.contourArea)
    return best + np.array([round(x0 * sx), round(y0 * sy)], np.int32)


def classify(cnt):
    """Name a contour from its convex hull.

    The hull is used because its perimeter isn't inflated by the pixel staircase
    and small bumps on the outline can't create extra corners.
    """
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_peri = cv2.arcLength(hull, True)
    if hull_area == 0 or hull_peri == 0:
        return "unknown"
    if area / hull_area < 0.9:
        # Concave outline (low solidity): a shape partly hidden behind another.
        return "partial"
    # Circularity 4*pi*A/P^2 is 1 for a circle, 0.91 for a regular hexagon and
    # 0.86 for a regular pentagon. Detected circles measure 0.99+ on their hull,
    # while a pentagon whose outline bulges on a weak edge reached ~0.93, so the
    # cut-off sits at 0.96. (Checking corners before circularity was tried: a
    # circle's hull can have unevenly spaced vertices that group into 3 "corners".)
    if 4 * np.pi * hull_area / (hull_peri * hull_peri) > 0.96:
        return "circle"
    corners = find_corners(hull, hull_peri)
    if len(corners) == 3:
        return "triangle"
    if len(corners) == 4:
        # Rectangle: both pairs of opposite sides parallel. The challenge's
        # trapezoid has legs leaning ~13 degrees each way, so they differ by ~26.
        return "rectangle" if opposite_sides_parallel(corners) else "trapezoid"
    if len(corners) == 5:
        return "pentagon"
    if len(corners) < 3:
        # A thin sliver: the visible piece of a shape split in two by another.
        return "partial"
    return f"{len(corners)}-gon"


def turning_angles(pts):
    """Absolute turn in degrees at each vertex of a closed polygon."""
    edges = np.roll(pts, -1, axis=0) - pts  # edge i runs from vertex i to i+1
    heading = np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))
    return np.abs((heading - np.roll(heading, 1) + 180) % 360 - 180)


def find_corners(hull, peri, short_edge=0.06, min_turn=35.0, collinear=10.0):
    """Locate the places where a convex outline turns sharply.

    A convex outline turns 360 degrees in total: 3 x 120 for a triangle,
    4 x 90 for a rectangle, 5 x 72 for a pentagon.
      1. Drop near-collinear vertices (turn < `collinear`). Otherwise a straight
         side broken into short pieces can merge the corners at both its ends.
      2. A pixelated corner is often bevelled into a few short edges, which
         splits its turn. Vertices joined by short edges (< `short_edge` of the
         perimeter) form a group, and their turns are added up.
      3. Each group turning more than `min_turn` degrees is one corner, placed
         where the long sides on either side of it would meet.

    (Simplifying the polygon with approxPolyDP was tried first; bevelled and
    rounded corners kept adding or dropping vertices.)
    """
    pts = hull.reshape(-1, 2).astype(np.float64)
    while len(pts) > 3:
        turns = turning_angles(pts)
        i = int(np.argmin(turns))
        if turns[i] >= collinear:
            break
        pts = np.delete(pts, i, axis=0)

    k = len(pts)
    edges = np.roll(pts, -1, axis=0) - pts
    long_edge = np.linalg.norm(edges, axis=1) >= short_edge * peri
    if k < 3 or not long_edge.any():
        return []
    turns = turning_angles(pts)

    corners, group_turn = [], 0.0
    start = (int(np.argmax(long_edge)) + 1) % k  # first vertex after a long edge
    entry = start                                # first vertex of the current group
    for j in range(k):
        i = (start + j) % k
        group_turn += turns[i]
        if long_edge[i]:  # the edge leaving this vertex is long, so the group ends here
            if group_turn > min_turn:
                corners.append(line_intersection(pts[entry - 1], pts[entry], pts[i], pts[(i + 1) % k]))
            group_turn, entry = 0.0, (i + 1) % k
    return corners


def line_intersection(a, b, c, d):
    """Point where line a-b meets line c-d (midpoint of b and c if parallel)."""
    r, s = b - a, d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return (b + c) / 2
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
    return a + t * r


def opposite_sides_parallel(corners, tol_deg=8.0):
    """True if both pairs of opposite sides of a quadrilateral are parallel."""
    p = np.asarray(corners)
    sides = [p[(i + 1) % 4] - p[i] for i in range(4)]

    def angle(u, v):
        cos = abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))
        return np.degrees(np.arccos(min(1.0, cos)))

    return angle(sides[0], sides[2]) < tol_deg and angle(sides[1], sides[3]) < tol_deg
