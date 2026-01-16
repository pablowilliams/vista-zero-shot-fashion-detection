"""
OWL-ViT Open-Vocabulary Object Detection Module
Project VISTA: Zero-Shot Fashion Object Detection

This module implements open-vocabulary object detection using OWL-ViT
(Vision Transformer for Open-World Localisation) for fashion item detection.
"""

import torch
from PIL import Image
from typing import List, Dict, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np

from transformers import OwlViTProcessor, OwlViTForObjectDetection


@dataclass
class Detection:
    """Represents a single object detection."""
    bbox: List[float]  # [x_min, y_min, x_max, y_max]
    label: str
    confidence: float
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)
    
    @property
    def centre(self) -> tuple:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    def to_coco_format(self, image_id: int, annotation_id: int, category_id: int) -> dict:
        """Convert to COCO annotation format."""
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [
                self.bbox[0], 
                self.bbox[1], 
                self.width, 
                self.height
            ],  # COCO uses [x, y, width, height]
            "area": self.area,
            "iscrowd": 0,
            "score": self.confidence
        }


class OWLViTDetector:
    """
    Open-vocabulary object detector using OWL-ViT.
    
    OWL-ViT enables text-conditioned object detection, allowing detection
    of any object class specified through natural language without requiring
    class-specific training.
    """
    
    def __init__(
        self, 
        model_name: str = "google/owlvit-base-patch32",
        device: Optional[str] = None
    ):
        """
        Initialise OWL-ViT detector.
        
        Args:
            model_name: Hugging Face model identifier
            device: Compute device (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading OWL-ViT model: {model_name}")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Default fashion queries
        self.fashion_queries = [
            "t-shirt", "shirt", "blouse", "sweater", "hoodie",
            "jacket", "coat", "blazer", "vest",
            "dress", "skirt", "shorts", "trousers", "jeans",
            "shoes", "sneakers", "boots", "heels", "sandals",
            "bag", "handbag", "backpack", "clutch",
            "watch", "sunglasses", "hat", "scarf", "belt",
            "necklace", "earrings", "bracelet", "ring"
        ]
    
    def detect(
        self,
        image: Image.Image,
        text_queries: Optional[List[str]] = None,
        threshold: float = 0.1,
        nms_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Perform open-vocabulary object detection.
        
        Args:
            image: PIL Image to process
            text_queries: Text descriptions of objects to detect
            threshold: Confidence threshold for detections
            nms_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of Detection objects
        """
        queries = text_queries or self.fashion_queries
        
        # Prepare inputs
        inputs = self.processor(
            text=[queries],  # Batch of 1
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.Tensor([[image.height, image.width]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]  # First (and only) image
        
        # Convert to Detection objects
        detections = []
        for score, label_idx, box in zip(
            results["scores"], 
            results["labels"], 
            results["boxes"]
        ):
            detections.append(Detection(
                bbox=box.cpu().tolist(),
                label=queries[label_idx],
                confidence=score.cpu().item()
            ))
        
        # Apply NMS
        if nms_threshold < 1.0:
            detections = self._apply_nms(detections, nms_threshold)
        
        return detections
    
    def _apply_nms(
        self, 
        detections: List[Detection], 
        iou_threshold: float
    ) -> List[Detection]:
        """
        Apply non-maximum suppression to filter overlapping detections.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: -x.confidence)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                det for det in detections
                if self._compute_iou(best.bbox, det.bbox) < iou_threshold
            ]
        
        return keep
    
    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def detect_with_scores(
        self,
        image: Image.Image,
        text_queries: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> Dict[str, List[Detection]]:
        """
        Detect objects and group by category.
        
        Args:
            image: PIL Image
            text_queries: Object categories to detect
            threshold: Confidence threshold
            
        Returns:
            Dictionary mapping categories to their detections
        """
        detections = self.detect(image, text_queries, threshold)
        
        grouped = {}
        for det in detections:
            if det.label not in grouped:
                grouped[det.label] = []
            grouped[det.label].append(det)
        
        return grouped
    
    def batch_detect(
        self,
        images: List[Image.Image],
        text_queries: Optional[List[str]] = None,
        threshold: float = 0.1,
        batch_size: int = 8
    ) -> List[List[Detection]]:
        """
        Batch detection for multiple images.
        
        Args:
            images: List of PIL Images
            text_queries: Object categories
            threshold: Confidence threshold
            batch_size: Processing batch size
            
        Returns:
            List of detection lists, one per image
        """
        queries = text_queries or self.fashion_queries
        all_detections = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Process batch
            inputs = self.processor(
                text=[queries] * len(batch_images),
                images=batch_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get target sizes for each image
            target_sizes = torch.Tensor([
                [img.height, img.width] for img in batch_images
            ]).to(self.device)
            
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )
            
            # Convert each result
            for j, result in enumerate(results):
                detections = []
                for score, label_idx, box in zip(
                    result["scores"],
                    result["labels"],
                    result["boxes"]
                ):
                    detections.append(Detection(
                        bbox=box.cpu().tolist(),
                        label=queries[label_idx],
                        confidence=score.cpu().item()
                    ))
                all_detections.append(detections)
        
        return all_detections


class DetectionFilter:
    """
    Applies geometric and semantic filtering to raw detections.
    """
    
    def __init__(
        self,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.95
    ):
        """
        Initialise detection filter.
        
        Args:
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            min_area_ratio: Minimum detection area as fraction of image
            max_area_ratio: Maximum detection area as fraction of image
        """
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
    
    def filter(
        self,
        detections: List[Detection],
        image_size: tuple
    ) -> List[Detection]:
        """
        Apply geometric filters to detections.
        
        Args:
            detections: List of Detection objects
            image_size: (width, height) of source image
            
        Returns:
            Filtered detections
        """
        image_area = image_size[0] * image_size[1]
        filtered = []
        
        for det in detections:
            # Aspect ratio check
            if not (self.min_aspect_ratio <= det.aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Area ratio check
            area_ratio = det.area / image_area
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue
            
            # Boundary check (detection should be within image)
            if det.bbox[0] < 0 or det.bbox[1] < 0:
                continue
            if det.bbox[2] > image_size[0] or det.bbox[3] > image_size[1]:
                continue
            
            filtered.append(det)
        
        return filtered


if __name__ == "__main__":
    # Example usage
    detector = OWLViTDetector()
    
    # Create test image
    test_image = Image.new("RGB", (640, 480), color=(200, 200, 200))
    
    # Detect objects
    detections = detector.detect(
        test_image,
        text_queries=["shirt", "dress", "shoes"],
        threshold=0.05
    )
    
    print(f"Found {len(detections)} detections:")
    for det in detections:
        print(f"  {det.label}: {det.confidence:.3f} at {det.bbox}")
