"""
Dataset Preparation & Annotation Generator.
Sets up the standard YOLO format folder structure:
  dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/
Can extract diverse frames from traffic videos and generate ground-truth
starter annotations for the 7 target classes:
  0: helmet
  1: no_helmet
  2: motorcycle
  3: person
  4: license_plate
  5: seatbelt
  6: no_seatbelt
"""

import os
import sys
import cv2
import random
import numpy as np
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO

from detection.rules import (
    is_rider_on_motorcycle,
    extract_head_crop,
    extract_driver_chest_crop
)


def init_dataset_dirs(base_dir="dataset"):
    splits = ["train", "valid"]
    for s in splits:
        os.makedirs(os.path.join(base_dir, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, s, "labels"), exist_ok=True)
    print(f"[*] Initialized YOLO directory structure in '{base_dir}/'.")


def extract_and_annotate_samples(
    video_path="input/sample3.mp4",
    num_frames=40,
    base_dir="dataset",
    train_ratio=0.80
):
    init_dataset_dirs(base_dir)

    print(f"[*] Extracting and annotating frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open video {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_video_frames // max(num_frames, 1))

    device = 0 if torch.cuda.is_available() else "cpu"
    base_model = YOLO("yolo11n.pt")

    extracted = 0
    frame_idx = 0

    while extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        h, w = frame.shape[:2]

        # Determine split (train vs valid)
        split = "train" if random.random() < train_ratio else "valid"
        img_name = f"traffic_frame_{extracted:04d}.jpg"
        lbl_name = f"traffic_frame_{extracted:04d}.txt"

        img_path = os.path.join(base_dir, split, "images", img_name)
        lbl_path = os.path.join(base_dir, split, "labels", lbl_name)

        # Run base detector to propose bounding boxes
        results = base_model(frame, conf=0.30, device=device, verbose=False)[0]

        annotations = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()

            motorcycles = [b for b, c in zip(boxes, classes) if c == 3]
            people = [b for b, c in zip(boxes, classes) if c == 0]
            cars = [b for b, c in zip(boxes, classes) if c in [2, 5, 7]]

            # 1. Annotate Persons & Motorcycles
            for m in motorcycles:
                # Class 2: motorcycle
                annotations.append((2, m))
                # Class 4: license_plate estimate for bike
                mb_w = m[2] - m[0]
                mb_h = m[3] - m[1]
                plate_box = [
                    int(m[0] + mb_w * 0.25),
                    int(m[1] + mb_h * 0.70),
                    int(m[2] - mb_w * 0.25),
                    int(m[3] - mb_h * 0.05)
                ]
                annotations.append((4, plate_box))

                # Check riders on bike
                mounted_riders = [p for p in people if is_rider_on_motorcycle(p, m)]
                for r in mounted_riders:
                    annotations.append((3, r))  # Class 3: person
                    # Extract head crop
                    _, h_coords = extract_head_crop(frame, r)
                    if h_coords[2] > h_coords[0] and h_coords[3] > h_coords[1]:
                        # Class 1: no_helmet (or class 0: helmet)
                        annotations.append((1, list(h_coords)))

            # 2. Annotate Cars, Seatbelts, Plates
            for c in cars:
                # We don't have vehicle in 7-class list, but we annotate seatbelts and plates:
                c_w = c[2] - c[0]
                c_h = c[3] - c[1]

                # Class 4: license_plate
                plate_box = [
                    int(c[0] + c_w * 0.25),
                    int(c[1] + c_h * 0.60),
                    int(c[2] - c_w * 0.25),
                    int(c[3] - c_h * 0.08)
                ]
                if plate_box[2] > plate_box[0] and plate_box[3] > plate_box[1]:
                    annotations.append((4, plate_box))

                # Class 6: no_seatbelt / seatbelt
                _, chest_coords = extract_driver_chest_crop(frame, c)
                if chest_coords[2] > chest_coords[0] and chest_coords[3] > chest_coords[1]:
                    annotations.append((6, list(chest_coords)))

        # Save image
        cv2.imwrite(img_path, frame)

        # Write YOLO txt format: <class_id> <x_center> <y_center> <width> <height>
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls_id, box in annotations:
                bx1, by1, bx2, by2 = box
                # Clamp to frame boundaries
                bx1 = max(0, min(w - 1, bx1))
                bx2 = max(0, min(w - 1, bx2))
                by1 = max(0, min(h - 1, by1))
                by2 = max(0, min(h - 1, by2))

                bw = bx2 - bx1
                bh = by2 - by1
                if bw > 8 and bh > 8:
                    x_center = (bx1 + bx2) / 2.0 / w
                    y_center = (by1 + by2) / 2.0 / h
                    norm_w = bw / float(w)
                    norm_h = bh / float(h)
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        extracted += 1

    cap.release()
    print(f"[*] Completed! Extracted and annotated {extracted} images in '{base_dir}/'.")


if __name__ == "__main__":
    extract_and_annotate_samples(
        video_path="input/sample3.mp4",
        num_frames=30,
        base_dir="dataset"
    )
