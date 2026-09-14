"""Part 4: 3D positions of shape centres from a single calibrated camera."""

import cv2
import numpy as np


def circle_radius_px(circle, cut=2.0):
    """Pixel radius of a detected circle, robust to part of it being hidden.

    Fits a circle to the outline points by least squares (Kasa fit: the circle
    equation x^2 + y^2 = 2ax + 2by + c is linear in a, b, c), in two stages:
      1. Three refits dropping points more than `cut` px *inside* the rim: the
         straight cut where another shape covers part of the circle.
      2. Two refits dropping points more than `cut` px off the rim either way,
         which also removes an outward bulge where the outline leaked into a
         touching shape (one frame read 233 in without this stage).
    Trimming both sides from the start fails: the first fit, pulled off-centre
    by the straight cut, would also discard genuine rim points.

    Two simpler estimates were tried first:
      - r = sqrt(area / pi): shrinks when part of the circle is hidden, which
        sent depth from ~250 to 268 in on those frames.
      - the minimum enclosing circle: robust to hiding, but extreme outline
        pixels inflate it by ~3% even on a whole circle (Z 243 instead of 250).
    """
    pts = circle.contour.reshape(-1, 2).astype(np.float64)
    keep = np.ones(len(pts), bool)
    r = np.sqrt(circle.area / np.pi)
    for stage in ("inside", "inside", "inside", "both", "both"):
        if keep.sum() < 5:
            break
        x, y = pts[keep, 0], pts[keep, 1]
        design = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        (a, b, c), *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
        r = np.sqrt(c + a * a + b * b)
        off_rim = np.hypot(pts[:, 0] - a, pts[:, 1] - b) - r  # negative = inside
        keep = off_rim > -cut if stage == "inside" else np.abs(off_rim) < cut
    return r

# Intrinsic matrix given in the challenge (focal lengths in pixels).
K_CHALLENGE = np.array([[2564.3186869, 0.0, 0.0],
                        [0.0, 2569.70273111, 0.0],
                        [0.0, 0.0, 1.0]])
CIRCLE_RADIUS_IN = 10.0
CALIB_WIDTH = 1920  # assumed: K was calibrated at the videos' resolution (1920x1080)


class PlanarLocalizer:
    """Back-projects shape centres into camera coordinates, in inches.

    Camera frame (OpenCV convention): X right, Y down, Z out of the lens.
    Pinhole model: a point (X, Y, Z) lands on pixel
        u = fx * X / Z + cx,    v = fy * Y / Z + cy.

    1. Depth from the circle. A circle of radius R at depth Z appears with a
       pixel radius r = f * R / Z, so Z = f * R / r. The pixel radius comes from
       the circle's area, r = sqrt(A / pi), which averages over every boundary
       pixel instead of trusting a few extreme ones.
    2. Flat surface. All shapes lie on one plane facing the camera, so every
       centre shares the circle's depth.
    3. Invert the projection: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy.
    """

    def __init__(self, K=K_CHALLENGE, circle_radius=CIRCLE_RADIUS_IN, principal_point="K",
                 max_jump=0.03, history=15, patience=5):
        self.K = np.asarray(K, float)
        self.circle_radius = circle_radius
        # The given K has cx = cy = 0, which puts the optical axis through the
        # top-left pixel. "K" uses it exactly as given; "center" assumes the
        # optical axis passes through the image centre, as in most real cameras.
        self.principal_point = principal_point
        self.depth = None  # last accepted depth, reused while the circle is hidden
        # Consistency gate (uses past frames only, so it still works on a stream):
        # a single-frame depth jump larger than `max_jump` relative to the median
        # of recent accepted depths is treated as a bad measurement, e.g. another
        # shape's corner poking out from under the circle. After `patience`
        # rejections in a row the new value is accepted, so a real change in
        # height still gets through.
        self.max_jump = max_jump
        self.history = history
        self.patience = patience
        self._recent = []
        self._rejected = 0

    def intrinsics(self, frame_width, frame_height):
        s = frame_width / CALIB_WIDTH  # rescale K if the frame isn't 1920 wide
        fx, fy = self.K[0, 0] * s, self.K[1, 1] * s
        if self.principal_point == "center":
            cx, cy = frame_width / 2, frame_height / 2
        else:
            cx, cy = self.K[0, 2] * s, self.K[1, 2] * s
        return fx, fy, cx, cy

    def _update_depth(self, z):
        if len(self._recent) >= 5:
            median = float(np.median(self._recent))
            if abs(z - median) > self.max_jump * median and self._rejected < self.patience:
                self._rejected += 1
                return  # keep the previous depth
        self._rejected = 0
        self.depth = z
        self._recent = (self._recent + [z])[-self.history:]

    def localize(self, detections, frame_width, frame_height):
        """Return an (X, Y, Z) tuple, or None, for each detection."""
        fx, fy, cx, cy = self.intrinsics(frame_width, frame_height)
        circles = [d for d in detections if d.label == "circle"]
        if circles:
            z = np.sqrt(fx * fy) * self.circle_radius / circle_radius_px(
                max(circles, key=lambda d: d.area))
            self._update_depth(z)
        if self.depth is None:
            return [None] * len(detections)
        z = self.depth
        return [((u - cx) * z / fx, (v - cy) * z / fy, z) for u, v in (d.center for d in detections)]
