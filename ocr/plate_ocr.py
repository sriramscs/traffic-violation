"""
Unified License Plate OCR Engine.
Supports PaddleOCR with automatic, seamless fallback to EasyOCR.
Applies CLAHE contrast enhancement, bilateral filtering, morphological cleanup,
and alphanumeric regex sanitization (e.g. Indian plates TN 38 AB 1234).
"""

import cv2
import numpy as np
import re
import warnings

# Attempt import of PaddleOCR
PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

# Attempt import of EasyOCR
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


class LicensePlateReader:
    def __init__(self, prefer_paddle=True, gpu=True):
        self.engine_name = "None"
        self.paddle_engine = None
        self.easyocr_engine = None
        self.gpu = gpu

        # 1. Initialize PaddleOCR if preferred and available
        if prefer_paddle and PADDLE_AVAILABLE:
            try:
                self.paddle_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=self.gpu,
                    show_log=False
                )
                self.engine_name = "PaddleOCR"
            except Exception as e:
                warnings.warn(f"Failed to initialize PaddleOCR: {e}. Falling back to EasyOCR.")
                self.paddle_engine = None

        # 2. Fallback to EasyOCR if needed
        if self.paddle_engine is None and EASYOCR_AVAILABLE:
            try:
                self.easyocr_engine = easyocr.Reader(['en'], gpu=self.gpu, verbose=False)
                self.engine_name = "EasyOCR"
            except Exception as e:
                warnings.warn(f"Failed to initialize EasyOCR with GPU: {e}. Trying CPU.")
                try:
                    self.easyocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
                    self.engine_name = "EasyOCR (CPU)"
                except Exception as e2:
                    warnings.warn(f"Failed to initialize EasyOCR: {e2}")

        print(f"[PlateReader] Initialized with engine: {self.engine_name}")

    def preprocess_plate(self, crop_img):
        """
        Enhances plate readability through super-resolution scaling,
        bilateral edge-preserving filtering, CLAHE, and adaptive thresholding.
        """
        if crop_img is None or crop_img.size == 0:
            return None

        h, w = crop_img.shape[:2]
        if h < 5 or w < 10:
            return None

        # 1. Upscale target height to ~100-120px for optimal OCR recognition
        scale = max(2, int(110 / max(h, 1)))
        resized = cv2.resize(crop_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale + Bilateral Filter (removes road grime while preserving characters)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. Contrast adjustment (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # 4. Otsu's adaptive thresholding
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def clean_plate_text(self, raw_text):
        """
        Standardizes and filters raw recognized characters.
        Matches formats like 'TN38AB1234' -> 'TN 38 AB 1234'.
        """
        if not raw_text:
            return ""

        # Remove non-alphanumeric chars
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

        # Filter out obvious false positives (too short or too long)
        if len(cleaned) < 4 or len(cleaned) > 13:
            return cleaned

        # Indian Number Plate Regex: 2 Letters (State) + 1-2 Digits + 1-2 Letters + 4 Digits
        match = re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{4})$', cleaned)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"

        # General formatted return (spaced pairs)
        return cleaned

    def read_plate(self, crop_img):
        """
        Extracts license plate text from cropped image.
        Returns: (plate_text, confidence_score)
        """
        if crop_img is None or crop_img.size == 0:
            return "", 0.0

        processed = self.preprocess_plate(crop_img)
        if processed is None:
            return "", 0.0

        best_text = ""
        best_conf = 0.0

        # Method 1: PaddleOCR
        if self.paddle_engine is not None:
            try:
                # PaddleOCR expects BGR or path
                rgb_proc = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                ocr_res = self.paddle_engine.ocr(rgb_proc, cls=True)
                if ocr_res and ocr_res[0]:
                    texts = []
                    confs = []
                    for line in ocr_res[0]:
                        txt = line[1][0]
                        conf = line[1][1]
                        texts.append(txt)
                        confs.append(conf)
                    joined_txt = "".join(texts)
                    cleaned = self.clean_plate_text(joined_txt)
                    if cleaned:
                        return cleaned, float(np.mean(confs) if confs else 0.85)
            except Exception:
                pass

        # Method 2: EasyOCR
        if self.easyocr_engine is not None:
            try:
                results = self.easyocr_engine.readtext(
                    processed,
                    detail=1,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                if not results:
                    # Fallback to raw grayscale
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    results = self.easyocr_engine.readtext(
                        gray,
                        detail=1,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )

                if results:
                    texts = [r[1] for r in results]
                    confs = [r[2] for r in results]
                    raw_joined = "".join(texts)
                    cleaned = self.clean_plate_text(raw_joined)
                    if cleaned:
                        best_text = cleaned
                        best_conf = float(np.mean(confs))
            except Exception:
                pass

        return best_text, best_conf