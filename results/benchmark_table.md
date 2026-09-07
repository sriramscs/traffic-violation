# Experimental Evaluation & Ablation Study

| Framework Architecture | Precision | Recall | F1-Score | mAP@0.5 | FPS | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv11 Baseline (Detection Only)** | 88.4% | 84.1% | 86.2% | 0.865 | 0.8 | 1206.2 ms |
| **Proposed (YOLOv11 + ViT + ByteTrack)** | 95.8% | 92.4% | 94.1% | 0.938 | 0.4 | 2853.8 ms |

*Note: Evaluated on test footage with NVIDIA GPU acceleration.*
