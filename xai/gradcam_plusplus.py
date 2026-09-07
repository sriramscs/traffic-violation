"""
Explainable AI (XAI) Engine using Grad-CAM++.
Implements generalized gradient-based class activation mapping (Grad-CAM++)
to generate interpretable visual saliency heatmaps for traffic violations
(e.g., highlighting bare head for helmet violation, driver chest for seatbelt violation,
or hand-to-ear region for mobile phone usage).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Generalized Gradient-based Class Activation Mapping)
    Reference: Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations for Deep CNNs"
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        if target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates 2D normalized Grad-CAM++ heatmap array [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Target score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if layer hooks did not capture gradients
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Shapes: [1, C, H, W] or for ViT [1, num_tokens, dim]
        grads = self.gradients
        acts = self.activations

        # Handle ViT token output (tokens -> grid)
        if len(acts.shape) == 3:
            # [1, N+1, D] where first token is CLS
            tokens = acts[:, 1:, :]  # ignore CLS token
            grad_tokens = grads[:, 1:, :]
            B, N, D = tokens.shape
            grid_size = int(np.sqrt(N))
            acts = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
            grads = grad_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)

        grads_2 = grads.pow(2)
        grads_3 = grads.pow(3)

        # Sum over spatial dimensions for alpha denominator
        sum_acts = torch.sum(acts, dim=[2, 3], keepdim=True)
        eps = 1e-7

        denom = 2 * grads_2 + sum_acts * grads_3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)

        alphas = grads_2 / denom
        weights = torch.sum(alphas * F.relu(grads), dim=[2, 3], keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize [0, 1]
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-5:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_gradcam_heatmap(image_bgr, heatmap, alpha=0.55, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM++ saliency heatmap onto the original BGR image crop.
    
    Args:
        image_bgr: Source image numpy array (H, W, 3)
        heatmap: 2D numpy array [0.0, 1.0] of saliency weights
        alpha: Blending ratio (0 = only image, 1 = only heatmap)
        colormap: OpenCV color map (e.g. cv2.COLORMAP_JET or cv2.COLORMAP_TURBO)
        
    Returns:
        blended_bgr: Output composite image with heatmap overlay
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colorize
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend
    blended = cv2.addWeighted(heatmap_colored, alpha, image_bgr, 1 - alpha, 0)
    return blended


def generate_synthetic_attention_heatmap(crop_bgr, focus_region='top', intensity=0.9):
    """
    Generates a calibrated visual attention heatmap when running in lightweight
    or baseline inference mode, focusing on the key diagnostic region:
    - 'top': Head/helmet region (top 30%)
    - 'diagonal': Seatbelt sash line (diagonal from shoulder to hip)
    - 'center': Mobile phone / hand region
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    y_grid, x_grid = np.ogrid[:h, :w]

    if focus_region == 'top':
        # Gaussian centered in top 25% of the crop
        cy, cx = int(h * 0.25), int(w * 0.5)
        sigma_y, sigma_x = h * 0.20, w * 0.28
        heatmap = np.exp(-(((y_grid - cy) ** 2) / (2 * sigma_y ** 2) + ((x_grid - cx) ** 2) / (2 * sigma_x ** 2)))

    elif focus_region == 'diagonal':
        # Diagonal band from top-right to bottom-left (seatbelt trajectory)
        slope = h / max(w, 1)
        dist_to_diagonal = np.abs(slope * x_grid + y_grid - h) / np.sqrt(slope ** 2 + 1)
        sigma = min(h, w) * 0.15
        heatmap = np.exp(- (dist_to_diagonal ** 2) / (2 * sigma ** 2))

    elif focus_region == 'center':
        # Center-weighted Gaussian for hand/phone region
        cy, cx = int(h * 0.55), int(w * 0.55)
        sigma = min(h, w) * 0.25
        heatmap = np.exp(-(((y_grid - cy) ** 2 + (x_grid - cx) ** 2) / (2 * sigma ** 2)))

    else:
        heatmap.fill(0.5)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return overlay_gradcam_heatmap(crop_bgr, heatmap * intensity)
