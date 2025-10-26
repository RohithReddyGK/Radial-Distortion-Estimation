# HELPER FUNCTIONS

import base64 

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode()

def draw_detected(img, corners, pattern):
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, True)
    return vis

def draw_inliers_outliers(img, points, inlier_mask):
    vis = img.copy()
    pts = points.reshape(-1,2).astype(int)
    for i, p in enumerate(pts):
        color = (0,255,0) if inlier_mask[i] else (0,0,255)
        cv2.circle(vis, tuple(p.tolist()), 4, color, -1)
    return vis

def draw_reprojection_arrows(img, detected_pts, reprojected_pts):
    vis = img.copy()
    det = detected_pts.reshape(-1,2)
    rep = reprojected_pts.reshape(-1,2)
    for (u,v),(ur,vr) in zip(det, rep):
        p1 = (int(round(u)), int(round(v)))
        p2 = (int(round(ur)), int(round(vr)))
        cv2.circle(vis, p1, 4, (0,255,0), -1)
        cv2.circle(vis, p2, 4, (0,0,255), -1)
        cv2.line(vis, p1, p2, (255,0,0), 1)
    return vis

def rms_error(a, b):
    a = np.asarray(a).reshape(-1,2)
    b = np.asarray(b).reshape(-1,2)
    dif = a - b
    sq = np.sum(dif**2, axis=1)
    mse = np.mean(sq)
    return float(np.sqrt(mse))


# RANSAC FUNCTION.
def ransac_pnp(objpts, imgpts, Kmat, distcoeff, iterations=500, sample_size=6, reproj_thresh=3.0):
    N = objpts.shape[0]
    best_mask = None
    best_count = 0
    best_r = None
    best_t = None
    rng = np.random.default_rng(123456)
    for it in range(iterations):
        idx = rng.choice(N, sample_size, replace=False)
        obj_s = objpts[idx].astype(np.float32)
        img_s = imgpts[idx].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_s, img_s, Kmat, distcoeff, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj_all, _ = cv2.projectPoints(objpts.astype(np.float32), rvec, tvec, Kmat, distcoeff)
        proj_all = proj_all.reshape(-1,2)
        dists = np.linalg.norm(proj_all - imgpts.reshape(-1,2), axis=1)
        mask = dists < reproj_thresh
        cnt = mask.sum()
        if cnt > best_count:
            best_count = cnt
            best_mask = mask.copy()
            best_r = rvec
            best_t = tvec
        if best_count >= 0.95 * N:
            break
    return best_mask, best_r, best_t, best_count


#FLASK API
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, numpy as np
import math
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Flask CV backend running."})

@app.route('/api/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    cols = int(request.form.get('cols', 7))
    rows = int(request.form.get('rows', 5))

    # --- Read Image ---
    img_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)

    steps = []

    # --- Step 1: Detect Corners ---
    found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not found:
        return jsonify({"error": "Checkerboard detection failed"}), 400

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    vis_detect = draw_detected(img, corners_subpix, pattern)
    steps.append({
        "title": "Step 1 — Detect Checkerboard Corners",
        "description": f"Detected corners: {len(corners_subpix)}",
        "explanation": "We need to find the 2D image coordinates of the inner checkerboard corners. These are the observed image points that will be matched to ideal 3D planar points (Z=0). We also refine coordinates to sub-pixel accuracy for better calibration.",
        "imagesBase64": [image_to_base64(vis_detect)],
        "table": []
    })

    # --- Step 2: Build 3D Object Points ---
    cols, rows = pattern
    objp = np.zeros((cols*rows, 3), np.float32)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, :2] = np.vstack((xs.ravel(), ys.ravel())).T
    steps.append({
        "title": "Step 2 — Build 3D Grid Points",
        "description": f"{objp.shape[0]} ideal 3D points created",
        "explanation": "The checkerboard is planar; we assign coordinates (X, Y, 0) to each inner corner in grid units. These are the 'true' 3D points we project into the image via our camera model.",
        "imagesBase64": [],
        "table": []
    })

    # --- Step 3: Initial Calibration ---
    objpoints = [objp.astype(np.float32)]
    imgpoints = [corners_subpix.astype(np.float32)]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    proj_init, _ = cv2.projectPoints(objp.astype(np.float32), rvecs[0], tvecs[0], K, dist)
    vis_init = draw_reprojection_arrows(img, corners_subpix, proj_init)

    # Format Camera Matrix as a readable 3x3 matrix
    K_matrix_str = "[\n" + "\n".join(
        ["  " + "   ".join([f"{val:10.4f}" for val in row]) for row in K.tolist()]
    ) + "\n]"

    # Format Distortion Coefficients as a single row
    dist_str = "[" + "   ".join([f"{val:10.4f}" for val in dist.flatten()]) + "]"

    steps.append({
        "title": "Step 3 — Initial Calibration",
        "description": f"Initial RMS: {ret:.4f}",
        "explanation": "Using the detected image points and the ideal 3D points, OpenCV's calibrateCamera estimates camera matrix and distortion coefficients.",
        "imagesBase64": [image_to_base64(vis_init)],
        "table": [{"Camera Matrix": K_matrix_str, "Distortion Coefficients": dist_str}]
    })

    # --- Step 4: RANSAC Inliers ---
    inlier_mask, rvec_ransac, tvec_ransac, best_count = ransac_pnp(objp, corners_subpix, K, dist)
    if inlier_mask is None:
        inlier_mask = np.ones(len(objp), dtype=bool)
        best_count = len(objp)
    vis_io = draw_inliers_outliers(img, corners_subpix, inlier_mask)
    steps.append({
        "title": "Step 4 — RANSAC Inlier Selection",
        "description": f"{best_count} inliers selected",
        "explanation": "Detections can contain outliers (wrong corners). We run a RANSAC-like process: repeatedly estimate a pose from a small random subset using solvePnP, then count how many detected points are well explained by that pose (within a reprojection threshold). The model with the most inliers is chosen. This yields a robust inlier set for final refinement.",
        "imagesBase64": [image_to_base64(vis_io)],
        "table": []
    })

    # --- Step 5: Refined Calibration ---
    obj_inliers = objp[inlier_mask].astype(np.float32)
    img_inliers = corners_subpix[inlier_mask].astype(np.float32)
    ret_refined, K_refined, dist_refined, rvecs_refined, tvecs_refined = cv2.calibrateCamera([obj_inliers], [img_inliers], gray.shape[::-1], None, None)
    proj_refined, _ = cv2.projectPoints(objp.astype(np.float32), rvecs_refined[0], tvecs_refined[0], K_refined, dist_refined)
    vis_refined = draw_reprojection_arrows(img, corners_subpix, proj_refined)

    # Format Camera Matrix as a readable 3x3 matrix using the refined values
    K_matrix_str = "[\n" + "\n".join(
        ["  " + "   ".join([f"{val:10.4f}" for val in row]) for row in K_refined.tolist()]
    ) + "\n]"

    # Format Distortion Coefficients as a single row
    dist_str = "[" + "   ".join([f"{val:10.4f}" for val in dist_refined.flatten()]) + "]"

    steps.append({
        "title": "Step 5 — Refined Calibration",
        "description": f"RMS after refinement: {ret_refined:.4f}",
        "explanation": "With outliers removed, calibrateCamera is run again using only the inlier correspondences. This refines K and distortion coefficients for a better final model.",
        "imagesBase64": [image_to_base64(vis_refined)],
        "table": [{"Camera Matrix": K_matrix_str, "Distortion Coefficients": dist_str}]
    })

    # --- Step 6: Undistort Image ---
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K_refined, dist_refined, (w,h), alpha=0.0, newImgSize=(w,h))
    undistorted_img = cv2.undistort(img, K_refined, dist_refined, None, newcameramtx)
    steps.append({
        "title": "Step 6 — Undistorted Image",
        "description": f"Image corrected for radial distortion",
        "explanation": "With the refined K and distortion coefficients, we can remap the entire image to a new undistorted view. We also compute where the ideal planar grid points land in the undistorted image (useful for visualization/reporting).",
        "imagesBase64": [image_to_base64(undistorted_img)],
        "table": []
    })


    # --- Step 7: Reproject undistorted grid & compute residuals ---
    proj_back, _ = cv2.projectPoints(objp.astype(np.float32), rvecs_refined[0], tvecs_refined[0], K_refined, dist_refined)
    proj_back = proj_back.reshape(-1,2)
    corners_2d = corners_subpix.reshape(-1,2)
    residuals = np.linalg.norm(corners_2d - proj_back, axis=1)

    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    max_res = float(np.max(residuals))

    # Prepare table for first 10 points
    table_rows = []
    for i in range(min(10, len(residuals))):
        table_rows.append({
            "u_detected": float(corners_2d[i,0]),
            "v_detected": float(corners_2d[i,1]),
            "u_reproj": float(proj_back[i,0]),
            "v_reproj": float(proj_back[i,1]),
            "residual_px": float(residuals[i])
        })

    # Draw reprojection arrows image
    vis_reproj = draw_reprojection_arrows(img, corners_subpix, proj_back)

    steps.append({
        "title": "Step 7 — Reproject undistorted grid back to the original distorted image & compute residuals",
        "description": f"Number of detected points: {len(corners_2d)}\n"
                    f"Mean residual (RMS-like): {mean_res:.4f} px\n"
                    f"Median residual: {median_res:.4f} px\n"
                    f"Max residual: {max_res:.4f} px",
        "explanation": "We validate the model by taking the ideal grid points, applying the full model (pose + intrinsics + distortion), "
                    "and comparing the predicted locations in the original image with the detected corner positions. "
                    "Residuals (per-point errors) and RMS are computed.",
        "imagesBase64": [image_to_base64(vis_reproj)],
        "table": table_rows
    })


    return jsonify({"steps": steps})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)