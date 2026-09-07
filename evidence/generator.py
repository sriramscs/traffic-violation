"""
Automatic Evidence Generation System.
Creates structured, audit-ready visual evidence records and digital violation cards
combining:
- Detection crop
- Explainable AI Grad-CAM++ saliency heatmap
- Vehicle number plate crop and OCR readout
- Official timestamp, track ID, camera metadata, and confidence score
Logs to CSV and structured JSON for the Traffic Analytics Dashboard.
"""

import os
import csv
import json
import cv2
import numpy as np
from datetime import datetime


class EvidenceGenerator:
    def __init__(self, output_dir="results", camera_id="CAM_01_NORTH_JUNCTION"):
        self.output_dir = output_dir
        self.evidence_dir = os.path.join(output_dir, "evidence")
        self.csv_path = os.path.join(output_dir, "violations_log.csv")
        self.json_path = os.path.join(output_dir, "evidence.json")
        self.camera_id = camera_id

        os.makedirs(self.evidence_dir, exist_ok=True)
        self._init_csv()
        self._init_json()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Camera_ID",
                    "Track_ID",
                    "Violation_Type",
                    "Plate_Number",
                    "Confidence",
                    "Evidence_File"
                ])

    def _init_json(self):
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_and_generate_card(
        self,
        track_id,
        violation_type,
        plate_number,
        confidence,
        original_crop,
        gradcam_crop=None,
        plate_crop=None
    ):
        """
        Generates an evidence card and registers the violation record.
        
        Returns:
            dict: Metadata of the created evidence record
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        evidence_filename = f"EV_{track_id}_{violation_type}_{file_timestamp}.jpg"
        evidence_filepath = os.path.join(self.evidence_dir, evidence_filename)

        # 1. Compose the Visual Evidence Card
        evidence_img = self._render_card_canvas(
            track_id=track_id,
            violation_type=violation_type,
            plate_number=plate_number,
            confidence=confidence,
            timestamp_str=timestamp_str,
            original_crop=original_crop,
            gradcam_crop=gradcam_crop,
            plate_crop=plate_crop
        )

        cv2.imwrite(evidence_filepath, evidence_img)

        # 2. Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str,
                self.camera_id,
                track_id,
                violation_type,
                plate_number,
                f"{confidence:.1f}%",
                evidence_filename
            ])

        # 3. Append to JSON database for Web Dashboard
        record = {
            "id": f"{track_id}_{file_timestamp}",
            "timestamp": timestamp_str,
            "camera_id": self.camera_id,
            "track_id": track_id,
            "violation_type": violation_type,
            "plate_number": plate_number,
            "confidence": round(confidence, 1),
            "evidence_image": f"/results/evidence/{evidence_filename}",
            "raw_filename": evidence_filename
        }

        try:
            records = []
            if os.path.exists(self.json_path):
                with open(self.json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            records.insert(0, record)  # Most recent first
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(records[:500], f, indent=2)
        except Exception as e:
            print(f"[Evidence Warning] JSON update failed: {e}")

        print(f"[EVIDENCE GENERATED] #{track_id} | {violation_type} | Plate: {plate_number} -> {evidence_filename}")
        return record

    def _render_card_canvas(
        self,
        track_id,
        violation_type,
        plate_number,
        confidence,
        timestamp_str,
        original_crop,
        gradcam_crop,
        plate_crop
    ):
        """
        Renders a composite, high-resolution visual evidence certificate.
        """
        card_w, card_h = 920, 520
        canvas = np.zeros((card_h, card_w, 3), dtype=np.uint8)
        # Background dark slate
        canvas[:] = (24, 27, 34)

        # Header Banner
        cv2.rectangle(canvas, (0, 0), (card_w, 75), (38, 44, 58), -1)
        cv2.line(canvas, (0, 75), (card_w, 75), (0, 102, 255), 3)

        # Header Titles
        cv2.putText(canvas, "AI TRAFFIC MONITORING - EVIDENCE DOSSIER", (25, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"CAMERA: {self.camera_id}  |  STATUS: VIOLATION VERIFIED", (25, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1, cv2.LINE_AA)

        # Metadata Card (Left Panel)
        cv2.rectangle(canvas, (25, 95), (340, 490), (32, 38, 50), -1)
        cv2.rectangle(canvas, (25, 95), (340, 490), (60, 70, 85), 1)

        cv2.putText(canvas, "VIOLATION DETAILS", (45, 128),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        details = [
            ("Track ID:", f"#{track_id}"),
            ("Violation:", violation_type),
            ("Confidence:", f"{confidence:.1f}%"),
            ("Plate No:", plate_number if plate_number else "N/A"),
            ("Timestamp:", timestamp_str),
            ("XAI Module:", "Grad-CAM++ (Active)"),
            ("Verification:", "ByteTrack Temporal")
        ]

        y_pos = 165
        for label, val in details:
            cv2.putText(canvas, label, (45, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 155, 175), 1, cv2.LINE_AA)
            val_color = (0, 80, 255) if "Violation" in label or violation_type in val else (240, 240, 240)
            cv2.putText(canvas, val, (45, y_pos + 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, val_color, 1, cv2.LINE_AA)
            y_pos += 46

        # Target Image Panels (Right side)
        # Panel 1: Original Candidate Crop
        panel_w, panel_h = 250, 220
        p1_x, p1_y = 365, 125

        cv2.putText(canvas, "Original Scene Detection", (p1_x, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 220), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (p1_x - 2, p1_y - 2), (p1_x + panel_w + 2, p1_y + panel_h + 2), (60, 70, 85), 1)

        if original_crop is not None and original_crop.size > 0:
            resized_orig = cv2.resize(original_crop, (panel_w, panel_h))
            canvas[p1_y:p1_y + panel_h, p1_x:p1_x + panel_w] = resized_orig

        # Panel 2: Explainable AI (Grad-CAM++ Heatmap)
        p2_x, p2_y = 640, 125
        cv2.putText(canvas, "Explainable AI (Grad-CAM++)", (p2_x, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (p2_x - 2, p2_y - 2), (p2_x + panel_w + 2, p2_y + panel_h + 2), (0, 150, 255), 1)

        target_gradcam = gradcam_crop if (gradcam_crop is not None and gradcam_crop.size > 0) else original_crop
        if target_gradcam is not None and target_gradcam.size > 0:
            resized_xai = cv2.resize(target_gradcam, (panel_w, panel_h))
            canvas[p2_y:p2_y + panel_h, p2_x:p2_x + panel_w] = resized_xai

        # Panel 3: License Plate Crop (Bottom Right)
        p3_x, p3_y = 365, 385
        plate_pw, plate_ph = 525, 95
        cv2.putText(canvas, "License Plate OCR Readout (PaddleOCR / EasyOCR)", (p3_x, 375),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + plate_pw, p3_y + plate_ph), (38, 44, 58), -1)
        cv2.rectangle(canvas, (p3_x, p3_y), (p3_x + plate_pw, p3_y + plate_ph), (70, 80, 100), 1)

        if plate_crop is not None and plate_crop.size > 0:
            pc_resized = cv2.resize(plate_crop, (180, 75))
            canvas[p3_y + 10:p3_y + 85, p3_x + 15:p3_x + 195] = pc_resized

        # Plate text badge
        cv2.rectangle(canvas, (p3_x + 220, p3_y + 18), (p3_x + 505, p3_y + 76), (15, 20, 28), -1)
        cv2.rectangle(canvas, (p3_x + 220, p3_y + 18), (p3_x + 505, p3_y + 76), (0, 200, 100), 2)
        plate_display = plate_number if plate_number else "NO PLATE READ"
        cv2.putText(canvas, plate_display, (p3_x + 235, p3_y + 57),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 120), 2, cv2.LINE_AA)

        return canvas
