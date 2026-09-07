"""
ByteTrack Multi-Object Tracking & Temporal Consistency Manager.
Maintains vehicle and rider identities across consecutive video frames.
Ensures temporal verification (violations must persist over N frames before confirmation),
eliminates duplicate alerts, associates license plates with vehicle IDs,
and computes trajectory dynamics (e.g., sudden lane changes).
"""

import time
from collections import deque
from datetime import datetime
import numpy as np


class TrackRecord:
    def __init__(self, track_id, cls_id, cls_name, bbox, fps=30):
        self.track_id = track_id
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.fps = fps

        self.first_seen = time.time()
        self.last_seen = time.time()
        self.total_frames = 1

        # Centroid trajectory for speed and lane dynamics
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.trajectory = deque([(cx, cy)], maxlen=fps * 2)

        # License Plate Association
        self.plate_number = "Searching Plate..."
        self.plate_confidence = 0.0

        # Temporal Violation Counters (violation_type -> count)
        self.violation_counts = {}
        self.confirmed_violations = set()
        self.logged_violations = set()  # Avoids duplicate CSV/evidence outputs

    def update_position(self, bbox):
        self.bbox = bbox
        self.last_seen = time.time()
        self.total_frames += 1

        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.trajectory.append((cx, cy))

    def update_plate(self, plate_text, conf=1.0):
        if plate_text and len(plate_text) >= 4:
            if self.plate_number == "Searching Plate..." or conf > self.plate_confidence:
                self.plate_number = plate_text
                self.plate_confidence = conf

    def vote_violation(self, violation_type, min_votes_required=3):
        """
        Temporal voting: violation must be observed across consecutive frames
        to avoid transient false positives.
        """
        self.violation_counts[violation_type] = self.violation_counts.get(violation_type, 0) + 1
        if self.violation_counts[violation_type] >= min_votes_required:
            self.confirmed_violations.add(violation_type)
            return True
        return False

    def is_logged(self, violation_type):
        return violation_type in self.logged_violations

    def mark_logged(self, violation_type):
        self.logged_violations.add(violation_type)

    def check_sudden_lane_change(self, px_per_sec_thresh=140):
        """
        Computes horizontal lateral velocity dx/dt across trajectory.
        """
        if len(self.trajectory) < (self.fps // 2):
            return False

        start_cx, _ = self.trajectory[0]
        curr_cx, _ = self.trajectory[-1]
        elapsed_sec = len(self.trajectory) / float(self.fps)

        if elapsed_sec <= 0:
            return False

        dx_velocity = abs(curr_cx - start_cx) / elapsed_sec
        return dx_velocity > px_per_sec_thresh


class TrafficTrackerManager:
    """
    Unified ByteTrack Tracking Manager for traffic violation pipelines.
    """
    def __init__(self, fps=30, persistence_frames=4):
        self.fps = fps
        self.persistence_frames = persistence_frames
        self.tracks = {}  # track_id -> TrackRecord

    def update(self, yolo_results, class_names):
        """
        Synchronizes Ultralytics ByteTrack outputs with stateful track records.
        """
        active_ids = set()
        current_detections = []

        if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = yolo_results[0].boxes.id.int().cpu().tolist()
            class_ids = yolo_results[0].boxes.cls.int().cpu().tolist()
            confs = yolo_results[0].boxes.conf.float().cpu().tolist()

            for box, t_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                active_ids.add(t_id)
                cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]

                if t_id not in self.tracks:
                    self.tracks[t_id] = TrackRecord(t_id, cls_id, cls_name, box, fps=self.fps)
                else:
                    self.tracks[t_id].update_position(box)

                current_detections.append({
                    'track_id': t_id,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'box': box,
                    'conf': conf,
                    'record': self.tracks[t_id]
                })

        # Cleanup stale tracks inactive for > 60 seconds
        now = time.time()
        stale_ids = [t_id for t_id, tr in self.tracks.items() if (now - tr.last_seen) > 60.0]
        for t_id in stale_ids:
            del self.tracks[t_id]

        return current_detections

    def get_track(self, track_id):
        return self.tracks.get(track_id)
