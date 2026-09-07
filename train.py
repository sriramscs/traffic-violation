"""
Training and Fine-Tuning Module for YOLOv11 and Vision Transformer (ViT).
Supports:
- YOLOv11 Multi-violation training on custom datasets (dataset/data.yaml)
- GPU automatic acceleration and mixed precision (AMP)
- Checkpoint export and validation metrics reporting
"""

import os
import sys
import argparse
import torch
from ultralytics import YOLO


def train_yolo(
    data_yaml="dataset/data.yaml",
    model_weights="yolo11n.pt",
    epochs=50,
    batch_size=16,
    imgsz=640,
    device=None,
    project="results/training",
    name="yolo11_traffic_violation"
):
    print("=" * 70)
    print(" YOLOv11 TRAFFIC VIOLATION MODEL TRAINING")
    print("=" * 70)

    # 1. Device check
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Dataset check
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset configuration file not found at '{data_yaml}'!")
        return

    print(f"[*] Dataset Config: {data_yaml}")
    print(f"[*] Initial Weights: {model_weights}")
    print(f"[*] Hyperparameters: Epochs={epochs}, Batch={batch_size}, ImageSize={imgsz}")

    # 3. Load YOLOv11 model
    model = YOLO(model_weights)

    # 4. Execute Training Loop
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            amp=True,  # Automatic Mixed Precision
            plots=True,
            save=True,
            verbose=True
        )

        print("\n" + "=" * 70)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print(f" Best Weights saved at: {os.path.join(project, name, 'weights', 'best.pt')}")
        print(f" Validation plots: {os.path.join(project, name)}")
        print("=" * 70)
        return results

    except Exception as e:
        print(f"[!] Training encountered an error: {e}")
        print("[*] Note: Ensure dataset images and annotations are placed in 'dataset/train' and 'dataset/valid'.")


def validate_yolo(weights_path="yolo11n.pt", data_yaml="dataset/data.yaml"):
    """
    Evaluates trained weights on validation split and prints mAP metrics.
    """
    print(f"[*] Validating weights '{weights_path}' on '{data_yaml}'...")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print(f"Validation mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 Traffic Violation Model")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--device", default=None, help="Device ('0', 'cpu')")
    parser.add_argument("--val-only", action="store_true", help="Run validation only")

    args = parser.parse_args()

    if args.val_only:
        validate_yolo(weights_path=args.weights, data_yaml=args.data)
    else:
        train_yolo(
            data_yaml=args.data,
            model_weights=args.weights,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
