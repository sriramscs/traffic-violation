"""
Integrated YOLOv11 Multi-Violation Detection Engine.
Combines:
- YOLOv11 real-time detection & ByteTrack tracking
- Spatial violation rules (Triple Riding, Helmet, Seatbelt, Phone)
- Vision Transformer (ViT) feature refinement
- Grad-CAM++ Explainable AI heatmaps
- License Plate OCR (PaddleOCR / EasyOCR)
- Automatic Evidence Generation
"""

import cv2
import numpy as np
from detection.rules import (
    is_rider_on_motorcycle,
    extract_head_crop,
    extract_driver_chest_crop,
    check_phone_near_person
)


class TrafficViolationDetector:
    def __init__(
        self,
        yolo_model,
        vit_classifier,
        plate_reader,
        tracker_manager,
        evidence_generator,
        min_temporal_votes=3
    ):
        self.model = yolo_model
        self.vit = vit_classifier
        self.ocr = plate_reader
        self.tracker = tracker_manager
        self.evidence = evidence_generator
        self.min_temporal_votes = min_temporal_votes

        # Target COCO classes for traffic monitoring:
        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 67: cell phone
        self.target_classes = [0, 2, 3, 5, 7, 67]

    def process_frame(self, frame, conf_thresh=0.28, iou_thresh=0.45):
        """
        Runs complete detection, tracking, violation analysis, XAI, and annotation on a frame.
        
        Returns:
            annotated_frame: Frame with bounding boxes, HUD, and violation markers
            stats: Dictionary of frame detection metrics
        """
        h, w = frame.shape[:2]

        # 1. Run YOLO Tracking (ByteTrack)
        results = self.model.track(
            source=frame,
            persist=True,
            classes=self.target_classes,
            conf=conf_thresh,
            iou=iou_thresh,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = self.tracker.update(results, self.model.names)

        # Categorize active detections
        motorcycles = [d for d in detections if d['class_id'] == 3]
        cars_and_trucks = [d for d in detections if d['class_id'] in [2, 5, 7]]
        people = [d for d in detections if d['class_id'] == 0]
        phones = [d for d in detections if d['class_id'] == 67]

        active_violations_in_frame = []

        # ==========================================================
        # 2. MOTORCYCLE VIOLATIONS: Triple Riding & Helmet
        # ==========================================================
        for bike in motorcycles:
            b_box = bike['box']
            b_tid = bike['track_id']
            b_record = bike['record']

            # Match seated riders
            matched_riders = [p for p in people if is_rider_on_motorcycle(p['box'], b_box)]
            rider_count = len(matched_riders)

            # --- Plate Extraction for Bike ---
            self._attempt_plate_reading(frame, b_box, b_record)

            # --- Check 1: Triple Riding ---
            if rider_count >= 3:
                is_confirmed = b_record.vote_violation("TRIPLE_RIDING", self.min_temporal_votes)
                active_violations_in_frame.append((b_tid, "TRIPLE_RIDING", b_box))

                if is_confirmed and not b_record.is_logged("TRIPLE_RIDING"):
                    b_record.mark_logged("TRIPLE_RIDING")
                    crop = frame[max(0, b_box[1]):min(h, b_box[3]), max(0, b_box[0]):min(w, b_box[2])]
                    xai_res = self.vit.classify_and_explain(crop, violation_type='helmet')
                    plate_crop = self._get_plate_crop(frame, b_box)

                    self.evidence.log_and_generate_card(
                        track_id=b_tid,
                        violation_type="TRIPLE_RIDING",
                        plate_number=b_record.plate_number,
                        confidence=94.5,
                        original_crop=crop,
                        gradcam_crop=xai_res['gradcam_img'],
                        plate_crop=plate_crop
                    )

            # --- Check 2: Helmet Violation for Each Rider ---
            for r_idx, rider in enumerate(matched_riders):
                r_box = rider['box']
                head_crop, head_coords = extract_head_crop(frame, r_box)

                if head_crop is not None and head_crop.size > 0:
                    xai_result = self.vit.classify_and_explain(head_crop, violation_type='helmet')

                    if xai_result['is_violation']:
                        is_confirmed = b_record.vote_violation("NO_HELMET", self.min_temporal_votes)
                        active_violations_in_frame.append((b_tid, "NO_HELMET", head_coords))

                        if is_confirmed and not b_record.is_logged("NO_HELMET"):
                            b_record.mark_logged("NO_HELMET")
                            plate_crop = self._get_plate_crop(frame, b_box)

                            self.evidence.log_and_generate_card(
                                track_id=b_tid,
                                violation_type="NO_HELMET",
                                plate_number=b_record.plate_number,
                                confidence=xai_result['confidence'],
                                original_crop=head_crop,
                                gradcam_crop=xai_result['gradcam_img'],
                                plate_crop=plate_crop
                            )

        # ==========================================================
        # 3. CAR & TRUCK VIOLATIONS: Seatbelt & Plate OCR
        # ==========================================================
        for car in cars_and_trucks:
            c_box = car['box']
            c_tid = car['track_id']
            c_record = car['record']

            # Read plate when car enters optimal resolution window
            self._attempt_plate_reading(frame, c_box, c_record)

            # --- Check 3: Seatbelt Violation ---
            # Analyze driver chest ROI when car is within clear view (bottom 70% of frame)
            if c_box[3] > (h * 0.35):
                chest_crop, chest_coords = extract_driver_chest_crop(frame, c_box)
                if chest_crop is not None and chest_crop.size > 0:
                    xai_result = self.vit.classify_and_explain(chest_crop, violation_type='seatbelt')

                    if xai_result['is_violation']:
                        is_confirmed = c_record.vote_violation("NO_SEATBELT", self.min_temporal_votes)
                        active_violations_in_frame.append((c_tid, "NO_SEATBELT", chest_coords))

                        if is_confirmed and not c_record.is_logged("NO_SEATBELT"):
                            c_record.mark_logged("NO_SEATBELT")
                            plate_crop = self._get_plate_crop(frame, c_box)

                            self.evidence.log_and_generate_card(
                                track_id=c_tid,
                                violation_type="NO_SEATBELT",
                                plate_number=c_record.plate_number,
                                confidence=xai_result['confidence'],
                                original_crop=chest_crop,
                                gradcam_crop=xai_result['gradcam_img'],
                                plate_crop=plate_crop
                            )

            # --- Check 4: Sudden Lane Change ---
            if c_record.check_sudden_lane_change(px_per_sec_thresh=140):
                is_confirmed = c_record.vote_violation("SUDDEN_LANE_CHANGE", self.min_temporal_votes)
                active_violations_in_frame.append((c_tid, "SUDDEN_LANE_CHANGE", c_box))

                if is_confirmed and not c_record.is_logged("SUDDEN_LANE_CHANGE"):
                    c_record.mark_logged("SUDDEN_LANE_CHANGE")
                    crop = frame[max(0, c_box[1]):min(h, c_box[3]), max(0, c_box[0]):min(w, c_box[2])]
                    plate_crop = self._get_plate_crop(frame, c_box)

                    self.evidence.log_and_generate_card(
                        track_id=c_tid,
                        violation_type="SUDDEN_LANE_CHANGE",
                        plate_number=c_record.plate_number,
                        confidence=92.0,
                        original_crop=crop,
                        gradcam_crop=self.vit.classify_and_explain(crop, 'seatbelt')['gradcam_img'],
                        plate_crop=plate_crop
                    )

        # ==========================================================
        # 4. MOBILE PHONE VIOLATIONS
        # ==========================================================
        for phone in phones:
            ph_box = phone['box']
            for p in people:
                if check_phone_near_person(ph_box, p['box']):
                    p_tid = p['track_id']
                    p_record = p['record']
                    is_confirmed = p_record.vote_violation("PHONE_USAGE", self.min_temporal_votes)
                    active_violations_in_frame.append((p_tid, "PHONE_USAGE", ph_box))

                    if is_confirmed and not p_record.is_logged("PHONE_USAGE"):
                        p_record.mark_logged("PHONE_USAGE")
                        crop = frame[max(0, ph_box[1]):min(h, ph_box[3]), max(0, ph_box[0]):min(w, ph_box[2])]
                        xai_result = self.vit.classify_and_explain(crop, violation_type='phone')

                        self.evidence.log_and_generate_card(
                            track_id=p_tid,
                            violation_type="PHONE_USAGE",
                            plate_number="N/A (Rider/Ped)",
                            confidence=xai_result['confidence'],
                            original_crop=crop,
                            gradcam_crop=xai_result['gradcam_img'],
                            plate_crop=crop
                        )

        # ==========================================================
        # 5. DRAW VISUAL ANNOTATIONS & ON-SCREEN HUD
        # ==========================================================
        annotated = self._render_annotations(frame, detections, active_violations_in_frame)

        stats = {
            'total_vehicles': len(cars_and_trucks) + len(motorcycles),
            'total_people': len(people),
            'active_violations': len(active_violations_in_frame),
            'tracks_active': len(self.tracker.tracks)
        }

        return annotated, stats

    def _attempt_plate_reading(self, frame, vehicle_box, track_record):
        """Attempts OCR plate reading on bottom zone with highest optical resolution."""
        if track_record.plate_number != "Searching Plate...":
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = vehicle_box

        # Try OCR if vehicle is in the bottom 55% of the frame
        if y2 > (h * 0.45):
            plate_crop = self._get_plate_crop(frame, vehicle_box)
            if plate_crop is not None and plate_crop.size > 0:
                plate_text, conf = self.ocr.read_plate(plate_crop)
                if plate_text:
                    track_record.update_plate(plate_text, conf)

    def _get_plate_crop(self, frame, vehicle_box):
        """Extracts the center-lower zone where plates are installed."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = vehicle_box
        box_w = x2 - x1
        box_h = y2 - y1

        p_x1 = max(0, int(x1 + box_w * 0.20))
        p_x2 = min(w, int(x2 - box_w * 0.20))
        p_y1 = max(0, int(y1 + box_h * 0.55))
        p_y2 = min(h, int(y2 - box_h * 0.05))

        if p_x2 > p_x1 and p_y2 > p_y1:
            return frame[p_y1:p_y2, p_x1:p_x2]
        return None

    def _render_annotations(self, frame, detections, violations):
        """Draws bounding boxes, tracking labels, violation tags, and HUD."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw all active tracks
        for det in detections:
            x1, y1, x2, y2 = det['box']
            tid = det['track_id']
            cname = det['class_name']
            record = det['record']
            plate = record.plate_number

            # Check if this track has an active violation
            has_violation = any(v[0] == tid for v in violations)
            color = (0, 0, 255) if has_violation else (0, 255, 120)

            # Bounding Box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label Badge
            label_text = f"ID #{tid} | {cname.upper()}"
            if record.cls_id in [2, 3, 5, 7]:
                label_text += f" | {plate}"

            cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + len(label_text) * 9, y1), color, -1)
            cv2.putText(canvas, label_text, (x1 + 4, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Highlight specific violation areas (Head for no-helmet, phone, etc.)
        for tid, vtype, vbox in violations:
            vx1, vy1, vx2, vy2 = vbox
            cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(canvas, f"VIOLATION: {vtype}", (vx1, max(20, vy1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # On-screen HUD Bar (Top)
        cv2.rectangle(canvas, (0, 0), (w, 35), (20, 24, 30), -1)
        cv2.line(canvas, (0, 35), (w, 35), (0, 102, 255), 2)

        hud_text = (
            f"AI TRAFFIC DETECTOR  |  TRACKS: {len(detections)}  |  "
            f"ACTIVE VIOLATIONS: {len(violations)}  |  "
            f"OCR: {self.ocr.engine_name}  |  "
            f"XAI: Grad-CAM++  |  "
            f"ViT: {'ENABLED' if self.vit.use_vit else 'DISABLED (BASELINE)'}"
        )
        cv2.putText(canvas, hud_text, (15, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

        return canvas