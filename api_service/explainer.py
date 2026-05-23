import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class GradCAM:
    """GradCAM implementation for model explainability"""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Class Activation Map
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        output[:, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()

        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam

class CastingExplainer:
    """Explainer for casting defect detection using GradCAM"""

    def __init__(self, model_pipeline, target_layer_name: str = 'layer4'):
        self.pipeline = model_pipeline

        # Find target layer for GradCAM
        self.target_layer = self._find_layer(model_pipeline.model, target_layer_name)
        if self.target_layer is None:
            logger.warning(f"Target layer {target_layer_name} not found, using last conv layer")
            self.target_layer = self._find_last_conv_layer(model_pipeline.model)

        self.gradcam = GradCAM(model_pipeline.model, self.target_layer)

    def _find_layer(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Find a layer by name"""
        for name, layer in model.named_modules():
            if name == layer_name:
                return layer
        return None

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Find the last convolutional layer"""
        last_conv = None
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
        return last_conv

    def explain(self, image: Image.Image, prediction: int, confidence: float) -> Dict[str, Any]:
        """
        Generate explanation for a prediction

        Args:
            image: PIL Image
            prediction: Model prediction (0=OK, 1=Defect)
            confidence: Prediction confidence

        Returns:
            Dictionary with explanation data
        """
        try:
            # Preprocess image same as model
            transform = self.pipeline.transform
            input_tensor = transform(image).unsqueeze(0).to(self.pipeline.device)

            # Generate CAM
            cam = self.gradcam.generate_cam(input_tensor, target_class=prediction)

            # Convert CAM to heatmap
            heatmap = self._cam_to_heatmap(cam)

            # Get regions of interest
            regions = self._analyze_regions(cam)

            explanation = {
                "prediction": "defect" if prediction == 1 else "ok",
                "confidence": float(confidence),
                "heatmap": heatmap.tolist(),  # Base64 encoded or array
                "regions_of_interest": regions,
                "explanation": self._generate_text_explanation(regions, prediction)
            }

            return explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {
                "error": "Explanation generation failed",
                "prediction": "defect" if prediction == 1 else "ok",
                "confidence": float(confidence)
            }

    def _cam_to_heatmap(self, cam: np.ndarray) -> np.ndarray:
        """Convert CAM to RGB heatmap"""
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return heatmap

    def _analyze_regions(self, cam: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze regions of high activation"""
        # Threshold CAM
        threshold = np.percentile(cam, 90)  # Top 10% activation
        high_activation = cam > threshold

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            high_activation.astype(np.uint8), connectivity=8
        )

        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            regions.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "confidence": float(np.mean(cam[y:y+h, x:x+w]))
            })

        # Sort by confidence
        regions.sort(key=lambda x: x["confidence"], reverse=True)

        return regions[:5]  # Top 5 regions

    def _generate_text_explanation(self, regions: List[Dict[str, Any]], prediction: int) -> str:
        """Generate human-readable explanation"""
        if prediction == 0:  # OK
            return "The model found no significant defect patterns in the image."

        if not regions:
            return "The model detected potential defects but couldn't localize them clearly."

        num_regions = len(regions)
        if num_regions == 1:
            return f"The model detected 1 potential defect region with high confidence."
        else:
            return f"The model detected {num_regions} potential defect regions, with the most significant one having {regions[0]['confidence']:.2%} activation."