"""Overlay drawing shared by the image and video scripts."""

import cv2


def draw_detections(frame, detections, positions=None):
    """Trace outlines, mark centres, and label each shape (with X, Y, Z if given)."""
    out = frame.copy()
    s = frame.shape[1] / 960  # scale line widths and text with resolution
    th = max(1, round(2 * s))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, d in enumerate(detections):
        cv2.drawContours(out, [d.contour], -1, (255, 255, 255), th)
        u, v = round(d.center[0]), round(d.center[1])
        cv2.circle(out, (u, v), round(5 * s), (255, 255, 255), -1)
        cv2.circle(out, (u, v), round(3 * s), (0, 0, 255), -1)

        lines = [d.label]
        if positions is not None and positions[i] is not None:
            x, y, z = positions[i]
            lines.append(f"X={x:.1f} Y={y:.1f} Z={z:.1f} in")
        else:
            lines.append(f"({u}, {v}) px")
        # Put the text right of the centre, or left of it if it would run off the frame.
        text_w = max(cv2.getTextSize(t, font, 0.5 * s, th)[0][0] for t in lines)
        x = u + round(10 * s)
        if x + text_w > frame.shape[1]:
            x = max(0, u - round(10 * s) - text_w)
        for j, text in enumerate(lines):
            org = (x, v + round((18 * j - 4) * s))
            # dark outline first so the text reads on white and yellow shapes
            cv2.putText(out, text, org, font, 0.5 * s, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(out, text, org, font, 0.5 * s, (255, 255, 255), th, cv2.LINE_AA)
    return out
