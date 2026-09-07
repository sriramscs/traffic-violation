"""
Master Entry Point for AI-Powered Explainable Traffic Violation Detection System.
Orchestrates:
1. YOLOv11 Real-time Object Detection
2. Vision Transformer (ViT) Feature Enhancement
3. ByteTrack Multi-Object Tracking & Consistency
4. Unified PaddleOCR / EasyOCR Plate Recognition
5. Grad-CAM++ Explainable AI Visualizations
6. Automatic Evidence Dossier Generation
"""

import os
import sys
import argparse
import time
import cv2
import torch
from ultralytics import YOLO

from detection.yolo_detection import TrafficViolationDetector
from models.vit_classifier import ViTViolationClassifier
from tracking.tracker import TrafficTrackerManager
from ocr.plate_ocr import LicensePlateReader
from evidence.generator import EvidenceGenerator


def run_pipeline(
    input_source="input/sample3.mp4",
    output_video="results/output_annotated.mp4",
    yolo_weights="yolo11n.pt",
    use_vit=True,
    headless=False,
    max_frames=None,
    force_cpu=False,
    camera_id="CAM_01_NORTH_JUNCTION"
):
    print("=" * 70)
    print(" AI-POWERED EXPLAINABLE TRAFFIC VIOLATION DETECTION SYSTEM")
    print(" YOLOv11 | Vision Transformer | ByteTrack | OCR | Grad-CAM++")
    print("=" * 70)

    use_gpu = torch.cuda.is_available() and not force_cpu and os.environ.get("FORCE_CPU") != "1"
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if use_gpu else 'CPU (GPU Disabled)'})")
    print(f"[*] Input Source: {input_source}")
    print(f"[*] Vision Transformer (ViT): {'ENABLED' if use_vit else 'DISABLED (Baseline mode)'}")

    # 1. Initialize YOLOv11 Detector
    if not os.path.exists(yolo_weights):
        print(f"[!] Model weights '{yolo_weights}' not found locally. Loading standard YOLO11n...")
        yolo_weights = "yolo11n.pt"
    print(f"[*] Loading YOLOv11 weights: {yolo_weights}...")
    model = YOLO(yolo_weights)

    # 2. Initialize Vision Transformer & Grad-CAM++
    vit_classifier = ViTViolationClassifier(device=device, use_vit=use_vit)

    # 3. Initialize License Plate OCR Engine
    plate_reader = LicensePlateReader(prefer_paddle=True, gpu=torch.cuda.is_available())

    # 4. Initialize ByteTrack Manager
    tracker_manager = TrafficTrackerManager(fps=30, persistence_frames=3)

    # 5. Initialize Evidence Generator
    evidence_generator = EvidenceGenerator(output_dir="results", camera_id=camera_id)

    # 6. Assemble Full Detection Engine
    detector = TrafficViolationDetector(
        yolo_model=model,
        vit_classifier=vit_classifier,
        plate_reader=plate_reader,
        tracker_manager=tracker_manager,
        evidence_generator=evidence_generator,
        min_temporal_votes=3
    )

    # Open Video Source
    cap = cv2.VideoCapture(int(input_source) if input_source.isdigit() else input_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {input_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Video Info: {width}x{height} @ {fps:.1f} FPS, Total Frames: {total_frames}")

    # Video Writer
    writer = None
    if output_video:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print("[*] Processing frames... Press 'q' in the display window to exit early.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                print(f"[*] Reached limit of {max_frames} frames.")
                break

            # Process frame through the AI pipeline
            annotated_frame, stats = detector.process_frame(frame)

            # Write to output file if requested
            if writer is not None:
                writer.write(annotated_frame)

            # Interactive Display
            if not headless:
                # Downscale preview if larger than 1080p for smooth display
                disp_frame = annotated_frame
                if width > 1280:
                    disp_scale = 1280.0 / width
                    disp_frame = cv2.resize(annotated_frame, (1280, int(height * disp_scale)))

                cv2.imshow("AI Traffic Violation Monitor", disp_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[*] User requested termination.")
                    break

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                curr_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}/{total_frames} | FPS: {curr_fps:.1f} | Active Tracks: {stats['tracks_active']} | Violations: {stats['active_violations']}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[*] Annotated video saved to: {output_video}")
        if not headless:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n" + "=" * 70)
    print(" PROCESSING COMPLETED")
    print(f" Total Frames: {frame_count}")
    print(f" Elapsed Time: {total_time:.2f} seconds")
    print(f" Average Speed: {avg_fps:.2f} FPS")
    print(f" Evidence Directory: results/evidence/")
    print(f" Violations CSV Log: results/violations_log.csv")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Traffic Violation Detection System")
    parser.add_argument("--input", default="input/sample3.mp4", help="Path to input video file or webcam index")
    parser.add_argument("--output", default="results/output_annotated.mp4", help="Path to save annotated video")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO model checkpoint path")
    parser.add_argument("--no-vit", action="store_true", help="Disable ViT (Run in YOLOv11 baseline mode)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (disable GPU)")
    parser.add_argument("--camera-id", default="CAM_01_NORTH_JUNCTION", help="Camera metadata tag")

    args = parser.parse_args()

    run_pipeline(
        input_source=args.input,
        output_video=args.output,
        yolo_weights=args.weights,
        use_vit=not args.no_vit,
        headless=args.headless,
        max_frames=args.max_frames,
        force_cpu=args.cpu,
        camera_id=args.camera_id
    )