from __future__ import annotations

import math

import cv2
import numpy as np


def compute_orb_features(image: np.ndarray, nfeatures=16000, mask=None, gray_masking=True):
    orb = cv2.ORB_create(nfeatures=nfeatures, WTA_K=4)
    all_keypoints = []
    all_descriptors = []
    # Work on 3-channel slice
    bgr = image[:, :, :3] if image.shape[2] >= 3 else image
    if gray_masking:
        gray_mask = (bgr[:, :, 0] == bgr[:, :, 1]) & (bgr[:, :, 1] == bgr[:, :, 2])
    for ch in cv2.split(bgr):
        if gray_masking:
            ch_copy = ch.copy()
            ch_copy[gray_mask] = 255
            kp, des = orb.detectAndCompute(ch_copy, mask)
        else:
            kp, des = orb.detectAndCompute(ch, mask)
        if des is not None:
            all_keypoints.extend(kp)
            all_descriptors.append(des)
    if not all_descriptors:
        return [], None
    return all_keypoints, np.vstack(all_descriptors)


def _worker_compute_orb(png_path: str, nfeatures: int):
    """Worker function for ProcessPoolExecutor — loads PNG, computes ORB."""
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    orb = cv2.ORB_create(nfeatures=nfeatures, WTA_K=4)
    bgr = image[:, :, :3]
    gray_mask = (bgr[:, :, 0] == bgr[:, :, 1]) & (bgr[:, :, 1] == bgr[:, :, 2])
    all_keypoints = []
    all_descriptors = []
    for ch in cv2.split(bgr):
        ch_copy = ch.copy()
        ch_copy[gray_mask] = 255
        kp, des = orb.detectAndCompute(ch_copy, None)
        if des is not None:
            all_keypoints.extend(kp)
            all_descriptors.append(des)
    if not all_descriptors:
        return None
    keypoints_pts = [kp.pt for kp in all_keypoints]
    descriptors = np.vstack(all_descriptors)
    h, w = image.shape[:2]
    return {"path": png_path, "keypoints": keypoints_pts, "descriptors": descriptors, "w": w, "h": h}


def decompose_affine(matrix: np.ndarray):
    """Decompose 2x3 similarity matrix into (tx, ty, rotation_deg, scale)."""
    a = matrix[0, 0]
    c = matrix[1, 0]
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    scale = math.sqrt(a * a + c * c)
    rotation_deg = math.degrees(math.atan2(c, a))
    return tx, ty, rotation_deg, scale


def matching_pixels_pct(data1: dict, data2: dict, matrix: np.ndarray,
                         bbox2_transformed: np.ndarray, tolerance=15) -> float:
    h1, w1 = data1["h"], data1["w"]
    h2, w2 = data2["h"], data2["w"]
    x_min = max(0, int(np.min(bbox2_transformed[:, 0])))
    y_min = max(0, int(np.min(bbox2_transformed[:, 1])))
    x_max = min(w1, int(np.max(bbox2_transformed[:, 0])))
    y_max = min(h1, int(np.max(bbox2_transformed[:, 1])))
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    w_roi = x_max - x_min
    h_roi = y_max - y_min
    M3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
    T = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float64)
    M_roi3 = T @ M3
    M_roi = M_roi3[:2, :]
    img1 = data1["image"]
    img2 = data2["image"]
    roi2_warped = cv2.warpAffine(img2, M_roi, (w_roi, h_roi),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
    mask_warped = cv2.warpAffine(mask2, M_roi, (w_roi, h_roi),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = mask_warped > 0
    roi1 = img1[y_min:y_max, x_min:x_max]
    min_h = min(roi1.shape[0], roi2_warped.shape[0])
    min_w = min(roi1.shape[1], roi2_warped.shape[1])
    roi1 = roi1[:min_h, :min_w]
    roi2_warped = roi2_warped[:min_h, :min_w]
    valid = valid[:min_h, :min_w]
    diff = np.abs(roi1.astype(np.int16) - roi2_warped.astype(np.int16))
    within_tol = np.all(diff <= tolerance, axis=-1)
    match_map = within_tol & valid
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        return 0.0
    return (int(np.count_nonzero(match_map)) / valid_pixels) * 100.0


def estimate_pairwise_transform(data1: dict, data2: dict):
    """
    data1, data2: dicts with 'keypoints', 'descriptors', 'image', 'w', 'h'
    Returns (matrix, overlap_ratios, pixel_match_pct, match_count) or (None, ...)
    """
    kp1 = data1["keypoints"]
    kp2 = data2["keypoints"]
    des1 = data1["descriptors"]
    des2 = data2["descriptors"]
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, (0.0, 0.0, 0.0), 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    top_matches = sorted_matches[:150]
    if len(top_matches) < 4:
        return None, (0.0, 0.0, 0.0), 0

    points1 = np.array([kp1[m.queryIdx] for m in top_matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx] for m in top_matches], dtype=np.float32)
    matrix, _inliers = cv2.estimateAffinePartial2D(points2, points1)
    if matrix is None:
        return None, (0.0, 0.0, 0.0), 0

    w1, h1 = data1["w"], data1["h"]
    w2, h2 = data2["w"], data2["h"]
    bbox1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    bbox2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
    bbox2_t = cv2.transform(bbox2.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    # Intersection area
    ix1 = max(np.min(bbox1[:, 0]), np.min(bbox2_t[:, 0]))
    iy1 = max(np.min(bbox1[:, 1]), np.min(bbox2_t[:, 1]))
    ix2 = min(np.max(bbox1[:, 0]), np.max(bbox2_t[:, 0]))
    iy2 = min(np.max(bbox1[:, 1]), np.max(bbox2_t[:, 1]))
    if ix2 > ix1 and iy2 > iy1:
        intersect = (ix2 - ix1) * (iy2 - iy1)
    else:
        intersect = 0.0

    overlap1 = intersect / (w1 * h1) if w1 * h1 > 0 else 0.0
    overlap2 = intersect / (w2 * h2) if w2 * h2 > 0 else 0.0
    pixel_pct = matching_pixels_pct(data1, data2, matrix, bbox2_t)
    return matrix, (overlap1, overlap2, pixel_pct), len(sorted_matches)


def auto_detect_placements(pieces: list):
    """
    Run pairwise ORB matching for initial placement guesses.
    Sets piece.x, piece.y, piece.rotation_deg, piece.scale, piece.is_matched.
    """
    if not pieces:
        return

    # Build data dicts from pieces (use display-res for speed)
    orb_data = []
    for piece in pieces:
        img = piece.get_cropped_image()
        # Use display pixmap resolution for speed
        h, w = img.shape[:2]
        ds = piece.display_scale
        if ds < 1.0:
            small = cv2.resize(img, (int(w * ds), int(h * ds)), interpolation=cv2.INTER_AREA)
        else:
            small = img.copy()
        kps, des = compute_orb_features(small)
        if des is None:
            kps, des = [], None
        orb_data.append({
            "image": small,
            "keypoints": [kp.pt for kp in kps],
            "descriptors": des,
            "w": small.shape[1],
            "h": small.shape[0],
            "ds": ds,
        })

    n = len(pieces)
    placed = [False] * n
    placed[0] = True
    pieces[0].x = pieces[0].get_cropped_image().shape[1] / 2.0
    pieces[0].y = pieces[0].get_cropped_image().shape[0] / 2.0
    pieces[0].rotation_deg = 0.0
    pieces[0].scale = 1.0
    pieces[0].is_matched = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not placed[i]:
                continue
            if orb_data[i]["descriptors"] is None:
                continue
            for j in range(n):
                if placed[j]:
                    continue
                if orb_data[j]["descriptors"] is None:
                    continue
                matrix, (ov1, ov2, pixel), match_count = estimate_pairwise_transform(
                    orb_data[i], orb_data[j]
                )
                if matrix is None:
                    continue
                overlap = max(ov1, ov2)
                if overlap < 0.05 or overlap > 0.9 or pixel < 75.0:
                    continue

                # matrix maps piece j coords -> piece i coords (display-res)
                # Convert to world coords accounting for display_scale and current placement
                ds_i = orb_data[i]["ds"]
                ds_j = orb_data[j]["ds"]
                tx, ty, rot, sc = decompose_affine(matrix)
                # Scale tx/ty from display-res to full-res
                tx_world = tx / ds_i
                ty_world = ty / ds_i
                # sc in display units: actual scale = sc * ds_j / ds_i
                sc_world = sc * ds_j / ds_i

                img_j = pieces[j].get_cropped_image()
                hj, wj = img_j.shape[:2]
                # Center of piece j in piece i's local coords
                cj_local_x = tx_world + (wj / 2.0) * sc_world * math.cos(math.radians(rot)) \
                              - (hj / 2.0) * sc_world * math.sin(math.radians(rot))
                cj_local_y = ty_world + (wj / 2.0) * sc_world * math.sin(math.radians(rot)) \
                              + (hj / 2.0) * sc_world * math.cos(math.radians(rot))

                # Piece i's center in world coords
                ci_world_x = pieces[i].x
                ci_world_y = pieces[i].y
                img_i = pieces[i].get_cropped_image()
                hi, wi = img_i.shape[:2]

                # Translate piece i local coords to world
                rot_i = math.radians(pieces[i].rotation_deg)
                cos_i = math.cos(rot_i)
                sin_i = math.sin(rot_i)
                s_i = pieces[i].scale
                world_x = ci_world_x + s_i * (cos_i * (cj_local_x - wi/2) - sin_i * (cj_local_y - hi/2))
                world_y = ci_world_y + s_i * (sin_i * (cj_local_x - wi/2) + cos_i * (cj_local_y - hi/2))

                pieces[j].x = world_x
                pieces[j].y = world_y
                pieces[j].rotation_deg = pieces[i].rotation_deg + rot
                pieces[j].scale = sc_world * pieces[i].scale
                pieces[j].is_matched = True
                placed[j] = True
                changed = True

    # Place unmatched pieces in a row below
    unmatched = [i for i in range(n) if not placed[i]]
    if unmatched:
        # Find bottom of all placed pieces
        max_y = 0.0
        max_h = 0.0
        for i in range(n):
            if placed[i]:
                bb = pieces[i].get_bounding_box()
                bottom = bb.bottom()
                h_piece = pieces[i].get_cropped_image().shape[0] * pieces[i].scale
                if bottom > max_y:
                    max_y = bottom
                    max_h = h_piece
        row_y = max_y + max_h * 0.1 + 100
        row_x = 0.0
        for i in unmatched:
            img = pieces[i].get_cropped_image()
            h, w = img.shape[:2]
            pieces[i].x = row_x + w / 2.0
            pieces[i].y = row_y + h / 2.0
            pieces[i].rotation_deg = 0.0
            pieces[i].scale = 1.0
            pieces[i].is_matched = False
            row_x += w + 50


def snap_piece_to_neighbors(piece: "PuzzlePiece", neighbors: list):
    """
    Refine piece placement using ORB feature matching against neighbors.
    Uses 50000 features; finds geometric overlap region; extracts ROIs.
    Modifies piece.x, piece.y, piece.rotation_deg, piece.scale in-place.
    Returns True if snap succeeded.
    """
    piece_img = piece.get_cropped_image()
    ph, pw = piece_img.shape[:2]

    best_matrix = None
    best_pixel = 0.0

    for neighbor in neighbors:
        if neighbor is piece:
            continue
        neighbor_img = neighbor.get_cropped_image()
        nh, nw = neighbor_img.shape[:2]

        # World bounding boxes
        p_bb = piece.get_bounding_box()
        n_bb = neighbor.get_bounding_box()

        # Geometric overlap in world coords
        inter = p_bb.intersected(n_bb)
        if inter.isEmpty() or inter.width() < 10 or inter.height() < 10:
            continue

        # Piece ROI: map inter from world to piece-local coords
        M_piece = piece.get_affine_matrix()
        M_piece_inv = cv2.invertAffineTransform(M_piece)

        inter_corners = np.array([
            [inter.left(), inter.top()],
            [inter.right(), inter.top()],
            [inter.right(), inter.bottom()],
            [inter.left(), inter.bottom()],
        ], dtype=np.float32).reshape(-1, 1, 2)

        piece_corners = cv2.transform(inter_corners, M_piece_inv).reshape(-1, 2)
        px1 = max(0, int(np.min(piece_corners[:, 0])))
        py1 = max(0, int(np.min(piece_corners[:, 1])))
        px2 = min(pw, int(np.max(piece_corners[:, 0])))
        py2 = min(ph, int(np.max(piece_corners[:, 1])))
        if px2 <= px1 or py2 <= py1:
            continue
        piece_roi = piece_img[py1:py2, px1:px2]

        # Neighbor ROI
        M_neighbor = neighbor.get_affine_matrix()
        M_neighbor_inv = cv2.invertAffineTransform(M_neighbor)
        neighbor_corners = cv2.transform(inter_corners, M_neighbor_inv).reshape(-1, 2)
        nx1 = max(0, int(np.min(neighbor_corners[:, 0])))
        ny1 = max(0, int(np.min(neighbor_corners[:, 1])))
        nx2 = min(nw, int(np.max(neighbor_corners[:, 0])))
        ny2 = min(nh, int(np.max(neighbor_corners[:, 1])))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        neighbor_roi = neighbor_img[ny1:ny2, nx1:nx2]

        # Use alpha channel as mask to only match in non-transparent regions
        piece_alpha = piece_roi[:, :, 3] if piece_roi.shape[2] == 4 else None
        neighbor_alpha = neighbor_roi[:, :, 3] if neighbor_roi.shape[2] == 4 else None

        # Compute features in overlap — no gray masking (matches stitch-images-realign)
        kp_p, des_p = compute_orb_features(piece_roi, nfeatures=50000, gray_masking=False, mask=piece_alpha)
        kp_n, des_n = compute_orb_features(neighbor_roi, nfeatures=50000, gray_masking=False, mask=neighbor_alpha)
        if des_p is None or des_n is None or len(kp_p) < 4 or len(kp_n) < 4:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des_p, des_n), key=lambda m: m.distance)[:200]
        if len(matches) < 4:
            continue

        src_pts = np.float32([kp_p[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_n[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # matrix_roi: maps piece_roi coords -> neighbor_roi coords
        matrix_roi, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if matrix_roi is None:
            continue

        # Convert ROI matrix to global: piece local -> neighbor local
        # P_neighbor_roi = M_roi * (P_piece - piece_roi_offset) + neighbor_roi_offset
        # P_neighbor = P_neighbor_roi + neighbor_roi_offset_in_local... no:
        # P_neighbor_local = M_roi * P_piece_roi + corr
        # P_piece_roi = P_piece_local - (px1, py1)
        # P_neighbor_local = M_roi * (P_piece_local - (px1, py1)) + (nx1, ny1)
        R = matrix_roi[:, :2]
        corr_piece = R @ np.array([px1, py1])
        neighbor_local_offset = np.array([nx1, ny1])
        matrix_piece_to_neighbor_local = matrix_roi.copy()
        matrix_piece_to_neighbor_local[:, 2] = matrix_roi[:, 2] - corr_piece + neighbor_local_offset

        # Now convert to world coords:
        # P_world = M_neighbor * P_neighbor_local
        # We want M_new such that piece center maps to new world position.
        # Actually we want the world transform for the piece:
        # P_world = M_neighbor * M_piece_to_neighbor_local * P_piece_local
        M_nb = neighbor.get_affine_matrix()
        M_nb_hom = np.vstack([M_nb, [0, 0, 1]])
        M_ptol_hom = np.vstack([matrix_piece_to_neighbor_local, [0, 0, 1]])
        M_new_world_hom = M_nb_hom @ M_ptol_hom
        M_new_world = M_new_world_hom[:2, :]

        # Validate with pixel match
        data_piece = {"image": piece_img, "w": pw, "h": ph}
        data_neighbor = {"image": neighbor_img, "w": nw, "h": nh}
        bbox_piece_in_neighbor = cv2.transform(
            np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32).reshape(-1, 1, 2),
            matrix_piece_to_neighbor_local
        ).reshape(-1, 2)
        pct = matching_pixels_pct(data_neighbor, data_piece, matrix_piece_to_neighbor_local,
                                   bbox_piece_in_neighbor)
        if pct > best_pixel:
            best_pixel = pct
            best_matrix = M_new_world

    if best_matrix is None or best_pixel < 0.5:
        return False

    # Decompose new world matrix to update piece params
    tx, ty, rot, sc = decompose_affine(best_matrix)
    img = piece.get_cropped_image()
    h, w = img.shape[:2]
    # The matrix: world = M_new_world * local
    # center in world = M_new_world * [w/2, h/2]
    cx_l, cy_l = w / 2.0, h / 2.0
    a, b = best_matrix[0, 0], best_matrix[0, 1]
    c2, d = best_matrix[1, 0], best_matrix[1, 1]
    tx2, ty2 = best_matrix[0, 2], best_matrix[1, 2]
    piece.x = a * cx_l + b * cy_l + tx2
    piece.y = c2 * cx_l + d * cy_l + ty2
    piece.scale = math.sqrt(a * a + c2 * c2)
    piece.rotation_deg = math.degrees(math.atan2(c2, a))
    return True
