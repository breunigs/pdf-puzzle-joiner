import os
import glob
import json
import subprocess
import re

import cv2
import numpy as np

from .priority import LOW_PRIO


def parse_page_range(text: str) -> list:
    """Parse '1-3,5,7-8' -> [1,2,3,5,7,8]"""
    pages = []
    text = text.strip()
    if not text or text.lower() == "all":
        return None
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def split_pdf_pages(pdf_path: str, work_dir: str, pages=None) -> list:
    """Split PDF into per-page PDFs. Returns list of (page_num_1indexed, pdf_path) tuples."""
    subprocess.run(
        [*LOW_PRIO, "pdfseparate", pdf_path, os.path.join(work_dir, "page_%04d.pdf")],
        check=True,
    )
    pdf_pages = sorted(glob.glob(os.path.join(work_dir, "page_*.pdf")))
    result = [(i + 1, p) for i, p in enumerate(pdf_pages)]
    if pages:
        result = [(pn, p) for pn, p in result if pn in pages]
    return result


def _convert_and_process_page(args):
    """Worker: convert single PDF page to PNG, compute crop rect, save uncropped to cache.

    Args:
        args: (pdf_page_path, page_num, idx, cache_path, autocrop_path)

    Returns:
        (cache_path, bgra_array, crop_rect_or_None, idx) or None
    """
    pdf_page_path, page_num, idx, cache_path, autocrop_path = args
    png_path = pdf_page_path.replace(".pdf", ".png")
    subprocess.run(
        [
            *LOW_PRIO, "convert", "-density", "300", "-background", "white",
            pdf_page_path, "-background", "white", "-alpha", "remove",
            "-trim", "-depth", "8", "-quality", "00", png_path,
        ],
        check=True,
    )
    img = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    crop_rect = compute_auto_crop_rect(img)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if cache_path:
        cv2.imwrite(cache_path, bgra)
        if autocrop_path:
            with open(autocrop_path, "w") as f:
                json.dump(crop_rect, f)
    return (cache_path or png_path, bgra, crop_rect, idx)


def pdf_page_info(pdf_path: str) -> tuple:
    """Returns (page_count, {page_num: (width_pts, height_pts)})."""
    page_count = 0
    page_sizes = {}
    try:
        # First pass: get page count
        result = subprocess.run(
            [*LOW_PRIO, "pdfinfo", pdf_path], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                page_count = int(line.split(":")[1].strip())
                break
        if page_count > 0:
            # Second pass: get per-page sizes
            result = subprocess.run(
                [*LOW_PRIO, "pdfinfo", "-f", "1", "-l", str(page_count), pdf_path],
                capture_output=True, text=True, check=True,
            )
            for line in result.stdout.splitlines():
                m = re.match(r"Page\s+(\d+)\s+size:\s+([\d.]+)\s+x\s+([\d.]+)", line)
                if m:
                    pn = int(m.group(1))
                    pw = float(m.group(2))
                    ph = float(m.group(3))
                    page_sizes[pn] = (pw, ph)
    except Exception:
        pass
    return page_count, page_sizes


# ---------------------------------------------------------------------------
# Border-cropping (ported from extract-largest-rectangle-between-lines.py)
# ---------------------------------------------------------------------------

def find_largest_empty_rectangle(horizontal_lines, vertical_lines, img_width, img_height):
    y_boundaries = sorted(set([0, img_height] + [line[1] for line in horizontal_lines]))
    x_boundaries = sorted(set([0, img_width] + [line[0] for line in vertical_lines]))

    max_area = 0
    best_rect = (0, 0, img_width, img_height)

    for i in range(len(x_boundaries)):
        for j in range(i + 1, len(x_boundaries)):
            x1, x2 = x_boundaries[i], x_boundaries[j]
            for k in range(len(y_boundaries)):
                for l in range(k + 1, len(y_boundaries)):
                    y1, y2 = y_boundaries[k], y_boundaries[l]

                    is_valid = True
                    for h_line in horizontal_lines:
                        h_x1, h_y, h_x2, _ = h_line
                        if y1 < h_y < y2 and max(x1, h_x1) < min(x2, h_x2):
                            is_valid = False
                            break
                    if not is_valid:
                        continue

                    for v_line in vertical_lines:
                        v_x, v_y1, _, v_y2 = v_line
                        if x1 < v_x < x2 and max(y1, v_y1) < min(y2, v_y2):
                            is_valid = False
                            break
                    if not is_valid:
                        continue

                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_rect = (x1, y1, x2, y2)

    return best_rect


def autocrop_image_rect(image: np.ndarray, tolerance=5) -> tuple:
    """Returns (x, y, w, h) of the content region, or full image rect if no border."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference = int(gray[0, 0])
    mask = np.abs(gray.astype(int) - reference) <= tolerance
    content_mask = ~mask
    coords = np.argwhere(content_mask)
    h, w = image.shape[:2]
    if coords.size == 0:
        return (0, 0, w, h)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def autocrop_image(image: np.ndarray, tolerance=5) -> np.ndarray:
    x, y, w, h = autocrop_image_rect(image, tolerance)
    return image[y:y+h, x:x+w]


def autocrop_border_lines(image: np.ndarray) -> np.ndarray:
    """
    Detect frame/border lines and crop to the largest inner rectangle.
    Returns the cropped BGR image, or the original if no crop found.
    """
    img_height, img_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    min_length = min(img_width, img_height) * 0.7
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_length, maxLineGap=1,
    )

    horizontal_line_coords = []
    vertical_line_coords = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten().tolist()
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if (abs(angle) < 1 or abs(angle - 180) < 1) and length >= 0.7 * img_width:
                horizontal_line_coords.append(
                    tuple(sorted([(x1, y1), (x2, y2)]))
                )
            elif (abs(angle - 90) < 1 or abs(angle + 90) < 1) and length >= 0.8 * img_height:
                vertical_line_coords.append(
                    tuple(sorted([(x1, y1), (x2, y2)], key=lambda p: p[1]))
                )

    # Re-scan margins with relaxed gap
    margin_w = int(img_width * 0.05)
    v_xs = set(line[0][0] for line in vertical_line_coords)
    has_left_frame = any(margin_w * 0.3 < x < margin_w * 2 for x in v_xs)
    has_right_frame = any(x > img_width * 0.95 for x in v_xs)

    for side, has_frame, x_offset in [
        ("left", has_left_frame, 0),
        ("right", has_right_frame, img_width - margin_w),
    ]:
        if has_frame:
            continue
        strip = edges[:, x_offset:x_offset + margin_w]
        extra_lines = cv2.HoughLinesP(
            strip, rho=1, theta=np.pi / 180, threshold=80,
            minLineLength=min_length, maxLineGap=10,
        )
        if extra_lines is None:
            continue
        for line in extra_lines:
            ex1, ey1, ex2, ey2 = line.flatten().tolist()
            ex1 += x_offset
            ex2 += x_offset
            angle = np.rad2deg(np.arctan2(ey2 - ey1, ex2 - ex1))
            length = np.sqrt((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2)
            if (abs(angle - 90) < 1 or abs(angle + 90) < 1) and length >= 0.8 * img_height:
                vertical_line_coords.append(
                    tuple(sorted([(ex1, ey1), (ex2, ey2)], key=lambda p: p[1]))
                )

    h_lines_for_crop = [(l[0][0], l[0][1], l[1][0], l[1][1]) for l in horizontal_line_coords]
    v_lines_for_crop = [(l[0][0], l[0][1], l[1][0], l[1][1]) for l in vertical_line_coords]

    x1, y1, x2, y2 = find_largest_empty_rectangle(
        h_lines_for_crop, v_lines_for_crop, img_width, img_height
    )

    if x1 == 0 and y1 == 0 and x2 == img_width and y2 == img_height:
        return image

    crop_x1, crop_y1 = x1 + 1, y1 + 1
    crop_x2, crop_y2 = x2, y2
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        return autocrop_image(cropped, tolerance=5)
    return image


# ---------------------------------------------------------------------------
# Legend removal (ported from hide-legende-in-pic.py)
# ---------------------------------------------------------------------------

def find_tall_verticals(image: np.ndarray) -> list:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_len = int(height * 0.7)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return []
    tall_xs = []
    for i in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[i].flatten().tolist()
        if abs(x2 - x1) < 5 and abs(y2 - y1) >= min_len:
            tall_xs.append((x1 + x2) // 2)
    if not tall_xs:
        return []
    tall_xs.sort()
    clusters = []
    for x in tall_xs:
        if clusters and abs(x - clusters[-1][-1]) < 15:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return sorted(int(np.median(c)) for c in clusters)


def find_legend_column(image: np.ndarray, tall_verticals: list):
    if len(tall_verticals) < 2:
        return None
    height, width = image.shape[:2]
    for i in range(len(tall_verticals)):
        for j in range(i + 1, len(tall_verticals)):
            left_x = tall_verticals[i]
            right_x = tall_verticals[j]
            gap = right_x - left_x
            gap_pct = gap / width
            if right_x > width * 0.85 and 0.05 < gap_pct < 0.35:
                return (left_x, width)
            if left_x < width * 0.15 and 0.05 < gap_pct < 0.35:
                return (0, right_x)
    return None


def find_corner_boxes(image: np.ndarray, margin_xs=None) -> list:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    binary = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    left_edges = [0]
    right_edges = [width]
    if margin_xs:
        for mx in margin_xs:
            if mx < width * 0.15:
                left_edges.append(mx)
            elif mx > width * 0.85:
                right_edges.append(mx)

    min_area = width * height * 0.005
    max_area = width * height * 0.15
    tol = 15
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        rect_area = bw * bh
        if rect_area == 0:
            continue
        fill_ratio = area / rect_area
        if fill_ratio < 0.90:
            continue
        near_left = any(abs(x - e) < tol for e in left_edges)
        near_right = any(abs((x + bw) - e) < tol for e in right_edges)
        near_top = y < tol
        near_bottom = y + bh > height - tol
        if not ((near_left or near_right) and (near_top or near_bottom)):
            continue
        if bw > width * 0.30 or bh > height * 0.30:
            continue
        boxes.append((x, y, bw, bh))
    return merge_boxes(boxes)


def merge_boxes(boxes: list) -> list:
    if len(boxes) <= 1:
        return boxes
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            x1, y1, w1, h1 = boxes[i]
            ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                x2, y2, w2, h2 = boxes[j]
                bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
                if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used.add(j)
                    merged = True
            new_boxes.append((ax1, ay1, ax2 - ax1, ay2 - ay1))
            used.add(i)
        boxes = new_boxes
    return boxes


def remove_legend(image: np.ndarray) -> np.ndarray:
    """Remove legend column and corner boxes from BGR image."""
    tall_verts = find_tall_verticals(image)
    legend_column = find_legend_column(image, tall_verts)
    corner_boxes = find_corner_boxes(image, margin_xs=tall_verts)

    if not legend_column and not corner_boxes:
        return image

    height, width = image.shape[:2]
    result = image.copy()

    if legend_column:
        lx_start, lx_end = legend_column
        if lx_end >= width - 5:
            result = result[:, :lx_start].copy()
            corner_boxes = [(x, y, w, h) for x, y, w, h in corner_boxes if x + w <= lx_start]
        elif lx_start <= 5:
            result = result[:, lx_end:].copy()
            corner_boxes = [(x - lx_end, y, w, h) for x, y, w, h in corner_boxes if x >= lx_end]
        else:
            result[:, lx_start:lx_end] = 255

    new_h, new_w = result.shape[:2]
    for (x, y, bw, bh) in corner_boxes:
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(new_w, x + bw + pad)
        y2 = min(new_h, y + bh + pad)
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 255

    return result


def compute_auto_crop_rect(image: np.ndarray) -> tuple:
    """Compute crop rect (x, y, w, h) that border-crop + legend-removal would produce.

    Runs detection on the image without modifying it. Returns rect in original
    image coordinates, or None if no crop is needed.
    """
    img_h, img_w = image.shape[:2]

    # --- Step 1: border detection (same logic as autocrop_border_lines) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    min_length = min(img_w, img_h) * 0.7
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_length, maxLineGap=1,
    )
    horizontal_line_coords = []
    vertical_line_coords = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten().tolist()
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if (abs(angle) < 1 or abs(angle - 180) < 1) and length >= 0.7 * img_w:
                horizontal_line_coords.append(tuple(sorted([(x1, y1), (x2, y2)])))
            elif (abs(angle - 90) < 1 or abs(angle + 90) < 1) and length >= 0.8 * img_h:
                vertical_line_coords.append(
                    tuple(sorted([(x1, y1), (x2, y2)], key=lambda p: p[1]))
                )

    margin_w = int(img_w * 0.05)
    v_xs = set(line[0][0] for line in vertical_line_coords)
    has_left_frame = any(margin_w * 0.3 < x < margin_w * 2 for x in v_xs)
    has_right_frame = any(x > img_w * 0.95 for x in v_xs)
    for side, has_frame, x_offset in [
        ("left", has_left_frame, 0),
        ("right", has_right_frame, img_w - margin_w),
    ]:
        if has_frame:
            continue
        strip = edges[:, x_offset:x_offset + margin_w]
        extra_lines = cv2.HoughLinesP(
            strip, rho=1, theta=np.pi / 180, threshold=80,
            minLineLength=min_length, maxLineGap=10,
        )
        if extra_lines is None:
            continue
        for line in extra_lines:
            ex1, ey1, ex2, ey2 = line.flatten().tolist()
            ex1 += x_offset
            ex2 += x_offset
            angle = np.rad2deg(np.arctan2(ey2 - ey1, ex2 - ex1))
            length = np.sqrt((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2)
            if (abs(angle - 90) < 1 or abs(angle + 90) < 1) and length >= 0.8 * img_h:
                vertical_line_coords.append(
                    tuple(sorted([(ex1, ey1), (ex2, ey2)], key=lambda p: p[1]))
                )

    h_lines = [(l[0][0], l[0][1], l[1][0], l[1][1]) for l in horizontal_line_coords]
    v_lines = [(l[0][0], l[0][1], l[1][0], l[1][1]) for l in vertical_line_coords]

    bx1, by1, bx2, by2 = find_largest_empty_rectangle(h_lines, v_lines, img_w, img_h)

    # Offset from origin (same +1 as autocrop_border_lines)
    if bx1 == 0 and by1 == 0 and bx2 == img_w and by2 == img_h:
        # No border lines found — try autocrop on full image
        ax, ay, aw, ah = autocrop_image_rect(image, tolerance=5)
        if ax == 0 and ay == 0 and aw == img_w and ah == img_h:
            bx1, by1 = 0, 0
            bw, bh = img_w, img_h
        else:
            bx1, by1, bw, bh = ax, ay, aw, ah
    else:
        crop_x1, crop_y1 = bx1 + 1, by1 + 1
        crop_x2, crop_y2 = bx2, by2
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            bx1, by1 = 0, 0
            bw, bh = img_w, img_h
        else:
            sub = image[crop_y1:crop_y2, crop_x1:crop_x2]
            ax, ay, aw, ah = autocrop_image_rect(sub, tolerance=5)
            bx1 = crop_x1 + ax
            by1 = crop_y1 + ay
            bw, bh = aw, ah

    # --- Step 2: legend detection on the cropped region ---
    if bw > 0 and bh > 0:
        sub = image[by1:by1+bh, bx1:bx1+bw]
        tall_verts = find_tall_verticals(sub)
        legend_col = find_legend_column(sub, tall_verts)
        if legend_col:
            lx_start, lx_end = legend_col
            sub_w = bw
            if lx_end >= sub_w - 5:
                # Right legend: crop from right
                bw = lx_start
            elif lx_start <= 5:
                # Left legend: crop from left
                bx1 += lx_end
                bw -= lx_end

    if bx1 == 0 and by1 == 0 and bw == img_w and bh == img_h:
        return None
    return (bx1, by1, bw, bh)
