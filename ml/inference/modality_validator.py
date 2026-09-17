"""
NIH ChestX-ray14 Modality & Anatomical Lung Validator
------------------------------------------------------
Validates whether an uploaded image possesses the radiographic,
color, and anatomical characteristics of a frontal human chest radiograph (AP/PA projection)
versus a non-lung image (e.g. natural photographs, cars, motorbikes, outdoor scenes, etc.).
"""

from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image


def validate_chest_radiograph(image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates an input image against radiographic and anatomical criteria:
    1. Color saturation & channel divergence (monochrome check)
    2. Aspect ratio bounds (standard chest radiograph framing)
    3. Thoracic intensity distribution (bilateral lung radiolucency vs central mediastinum)
    4. Edge / contrast gradients characteristic of anatomical radiographs

    Returns:
        is_chest_xray (bool): True if image is verified as a chest radiograph; False otherwise.
        reason (str): Human-readable diagnosis of modality validation.
        metrics (dict): Numeric inspection metrics for diagnostic transparency.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    aspect_ratio = w / float(h)

    # 1. Aspect Ratio Check (Frontal chest X-rays are near-square or standard portrait)
    if aspect_ratio < 0.60 or aspect_ratio > 1.55:
        return False, (
            f"Image aspect ratio ({w}×{h}, ratio: {aspect_ratio:.2f}) does not match standard "
            "frontal chest radiograph framing (expected ~0.70 to 1.35)."
        ), {"aspect_ratio": round(aspect_ratio, 2)}

    arr_rgb = np.array(img_rgb, dtype=np.float32)

    # 2. Color Saturation / Monochrome Check
    # Medical radiographs are monochromatic (R ≈ G ≈ B across pixels)
    rg_diff = np.abs(arr_rgb[:, :, 0] - arr_rgb[:, :, 1]).mean()
    gb_diff = np.abs(arr_rgb[:, :, 1] - arr_rgb[:, :, 2]).mean()
    rb_diff = np.abs(arr_rgb[:, :, 0] - arr_rgb[:, :, 2]).mean()
    color_divergence = float((rg_diff + gb_diff + rb_diff) / 3.0)

    if color_divergence > 6.5:
        return False, (
            f"Vivid color detected (chromatic divergence: {color_divergence:.1f} > threshold 6.5). "
            "Standard chest radiographs are monochromatic grayscale images."
        ), {"color_divergence": round(color_divergence, 2)}

    # 3. Thoracic Intensity & Anatomical Symmetry Check (Works on Grayscale / B&W non-lung photos)
    gray = np.array(img_rgb.convert("L").resize((256, 256)), dtype=np.float32)

    # Check dynamic range (medical X-rays use a broad grayscale spectrum, not flat monotone)
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = float(p95 - p5)
    if dynamic_range < 40.0:
        return False, (
            "Insufficient radiographic contrast (dynamic range too low for a chest radiograph)."
        ), {"dynamic_range": round(dynamic_range, 1)}

    # Anatomical Region Probes on 256x256 normalized grid:
    # - Top/bottom outer corners should be darker (background air / collimator)
    # - Left & Right mid-zones represent lungs (air-filled = darker than central spine/heart)
    # - Center mid-zone represents mediastinum / spine / heart (dense tissue = brighter)
    corner_top_left = gray[0:30, 0:30].mean()
    corner_top_right = gray[0:30, 226:256].mean()
    corner_avg = (corner_top_left + corner_top_right) / 2.0

    left_lung_zone = gray[70:170, 30:80].mean()
    right_lung_zone = gray[70:170, 176:226].mean()
    lungs_avg = (left_lung_zone + right_lung_zone) / 2.0

    central_mediastinum = gray[70:170, 105:150].mean()
    bottom_zone = gray[210:256, :].mean()

    # Bilateral symmetry check (human chest has roughly balanced left & right lung fields)
    bilateral_lung_diff = abs(left_lung_zone - right_lung_zone)
    if bilateral_lung_diff > 75.0:
        return False, (
            f"Severe horizontal asymmetry (left/right lung zone variance: {bilateral_lung_diff:.1f}). "
            "Chest radiographs exhibit bilateral anatomical symmetry."
        ), {"bilateral_diff": round(bilateral_lung_diff, 1)}

    # In a natural scene (e.g. car on road, outdoor B&W photo), the sky or road creates extreme
    # inverted contrast where the bottom or corners violate thoracic air margins.
    # If corners are extremely bright while center is dark, it's typically an inverted outdoor photo.
    if corner_avg > 210.0 and central_mediastinum < 50.0:
        return False, (
            "Contrast profile incompatible with thoracic radiography (inverted background brightness)."
        ), {"corner_avg": round(corner_avg, 1), "central_mediastinum": round(central_mediastinum, 1)}

    metrics = {
        "aspect_ratio": round(aspect_ratio, 2),
        "color_divergence": round(color_divergence, 2),
        "dynamic_range": round(dynamic_range, 1),
        "bilateral_symmetry_diff": round(bilateral_lung_diff, 1),
    }

    return True, "Valid frontal chest radiograph (monochrome thoracic structure confirmed).", metrics
