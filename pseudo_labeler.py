"""
Pseudo-Label Generation Pipeline
Project VISTA: Zero-Shot Fashion Object Detection

This module implements the complete pseudo-labelling pipeline that
generates high-quality bounding box annotations from zero-shot detections.
"""

import os
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

from clip_classifier import CLIPClassifier
from owl_detector import OWLViTDetector, Detection, DetectionFilter


@dataclass
class PseudoLabel:
    """Represents a validated pseudo-label."""
    image_id: str
    bbox: List[float]
    label: str
    confidence: float
    clip_score: float
    ensemble_score: float
    
    def to_coco_annotation(
        self, 
        annotation_id: int, 
        image_id_int: int, 
        category_id: int
    ) -> dict:
        """Convert to COCO annotation format."""
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        return {
            "id": annotation_id,
            "image_id": image_id_int,
            "category_id": category_id,
            "bbox": [self.bbox[0], self.bbox[1], width, height],
            "area": width * height,
            "iscrowd": 0,
            "attributes": {
                "confidence": self.confidence,
                "clip_score": self.clip_score,
                "ensemble_score": self.ensemble_score
            }
        }


class PseudoLabelGenerator:
    """
    Generates pseudo-labels using vision-language model ensemble.
    
    Pipeline stages:
    1. OWL-ViT detection (generates candidate boxes)
    2. Confidence thresholding (filters low-confidence detections)
    3. CLIP verification (validates category assignment)
    4. Ensemble scoring (combines multiple model scores)
    5. Geometric filtering (removes implausible boxes)
    """
    
    def __init__(
        self,
        primary_threshold: float = 0.35,
        secondary_threshold: float = 0.20,
        ensemble_threshold: float = 0.50,
        device: Optional[str] = None
    ):
        """
        Initialise pseudo-label generator.
        
        Args:
            primary_threshold: High-confidence detection threshold
            secondary_threshold: Candidate pool threshold
            ensemble_threshold: Minimum ensemble score for validation
            device: Compute device
        """
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.ensemble_threshold = ensemble_threshold
        
        # Initialise models
        print("Initialising CLIP classifier...")
        self.clip_classifier = CLIPClassifier(device=device)
        
        print("Initialising OWL-ViT detector...")
        self.owl_detector = OWLViTDetector(device=device)
        
        # Geometric filter
        self.geo_filter = DetectionFilter(
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0,
            min_area_ratio=0.01,
            max_area_ratio=0.95
        )
        
        # Category mapping for COCO format
        self.category_mapping = self._build_category_mapping()
    
    def _build_category_mapping(self) -> Dict[str, int]:
        """Build category name to ID mapping."""
        categories = [
            "t-shirt", "shirt", "blouse", "sweater", "hoodie",
            "jacket", "coat", "blazer", "vest",
            "dress", "skirt", "shorts", "trousers", "jeans",
            "shoes", "sneakers", "boots", "heels", "sandals",
            "bag", "handbag", "backpack", "clutch",
            "watch", "sunglasses", "hat", "scarf", "belt",
            "necklace", "earrings", "bracelet", "ring"
        ]
        return {cat: i + 1 for i, cat in enumerate(categories)}
    
    def generate_for_image(
        self,
        image: Image.Image,
        image_id: str
    ) -> List[PseudoLabel]:
        """
        Generate pseudo-labels for a single image.
        
        Args:
            image: PIL Image
            image_id: Unique identifier for the image
            
        Returns:
            List of validated PseudoLabel objects
        """
        # Stage 1: OWL-ViT detection
        raw_detections = self.owl_detector.detect(
            image,
            threshold=self.secondary_threshold,
            nms_threshold=0.5
        )
        
        if not raw_detections:
            return []
        
        # Stage 2: Confidence-based partitioning
        high_conf = [d for d in raw_detections if d.confidence >= self.primary_threshold]
        medium_conf = [
            d for d in raw_detections 
            if self.secondary_threshold <= d.confidence < self.primary_threshold
        ]
        
        # Stage 3 and 4: CLIP verification and ensemble scoring
        validated_labels = []
        
        # High-confidence detections: direct validation
        for det in high_conf:
            clip_score = self._verify_with_clip(image, det)
            ensemble_score = 0.6 * clip_score + 0.4 * det.confidence
            
            validated_labels.append(PseudoLabel(
                image_id=image_id,
                bbox=det.bbox,
                label=det.label,
                confidence=det.confidence,
                clip_score=clip_score,
                ensemble_score=ensemble_score
            ))
        
        # Medium-confidence: require ensemble verification
        for det in medium_conf:
            clip_score = self._verify_with_clip(image, det)
            ensemble_score = 0.6 * clip_score + 0.4 * det.confidence
            
            if ensemble_score >= self.ensemble_threshold:
                validated_labels.append(PseudoLabel(
                    image_id=image_id,
                    bbox=det.bbox,
                    label=det.label,
                    confidence=det.confidence,
                    clip_score=clip_score,
                    ensemble_score=ensemble_score
                ))
        
        # Stage 5: Geometric filtering
        filtered_labels = self._apply_geometric_filter(
            validated_labels, 
            (image.width, image.height)
        )
        
        return filtered_labels
    
    def _verify_with_clip(self, image: Image.Image, detection: Detection) -> float:
        """
        Verify detection using CLIP similarity.
        
        Args:
            image: Source image
            detection: Detection to verify
            
        Returns:
            CLIP similarity score
        """
        # Crop detected region
        crop = image.crop([
            max(0, int(detection.bbox[0])),
            max(0, int(detection.bbox[1])),
            min(image.width, int(detection.bbox[2])),
            min(image.height, int(detection.bbox[3]))
        ])
        
        # Ensure minimum size
        if crop.width < 10 or crop.height < 10:
            return 0.0
        
        # Get CLIP similarity
        return self.clip_classifier.compute_similarity(crop, detection.label)
    
    def _apply_geometric_filter(
        self,
        labels: List[PseudoLabel],
        image_size: Tuple[int, int]
    ) -> List[PseudoLabel]:
        """
        Apply geometric filtering to pseudo-labels.
        
        Args:
            labels: List of pseudo-labels
            image_size: (width, height) of source image
            
        Returns:
            Filtered pseudo-labels
        """
        image_area = image_size[0] * image_size[1]
        filtered = []
        
        for label in labels:
            width = label.bbox[2] - label.bbox[0]
            height = label.bbox[3] - label.bbox[1]
            aspect_ratio = width / max(height, 1e-6)
            area = width * height
            area_ratio = area / image_area
            
            # Check constraints
            if not (0.2 <= aspect_ratio <= 5.0):
                continue
            if not (0.01 <= area_ratio <= 0.95):
                continue
            
            filtered.append(label)
        
        return filtered
    
    def generate_dataset(
        self,
        image_dir: str,
        output_dir: str,
        max_images: Optional[int] = None
    ) -> Dict:
        """
        Generate pseudo-labels for an entire dataset.
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory for pseudo-labels
            max_images: Maximum number of images to process
            
        Returns:
            COCO-format annotation dictionary
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if max_images:
            image_files = image_files[:max_images]
        
        # Initialise COCO structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": id, "name": name} 
                for name, id in self.category_mapping.items()
            ]
        }
        
        annotation_id = 1
        
        # Process images
        for img_idx, img_path in enumerate(tqdm(image_files, desc="Generating pseudo-labels")):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            image_id = img_path.stem
            image_id_int = img_idx + 1
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id_int,
                "file_name": img_path.name,
                "width": image.width,
                "height": image.height
            })
            
            # Generate pseudo-labels
            pseudo_labels = self.generate_for_image(image, image_id)
            
            # Add annotations
            for label in pseudo_labels:
                category_id = self.category_mapping.get(label.label, 1)
                annotation = label.to_coco_annotation(
                    annotation_id, 
                    image_id_int, 
                    category_id
                )
                coco_data["annotations"].append(annotation)
                annotation_id += 1
        
        # Save COCO annotations
        output_path = output_dir / "pseudo_labels.json"
        with open(output_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\nGenerated {len(coco_data['annotations'])} pseudo-labels for {len(coco_data['images'])} images")
        print(f"Saved to {output_path}")
        
        # Generate statistics
        self._generate_statistics(coco_data, output_dir)
        
        return coco_data
    
    def _generate_statistics(self, coco_data: Dict, output_dir: Path) -> None:
        """Generate and save dataset statistics."""
        stats = {
            "total_images": len(coco_data["images"]),
            "total_annotations": len(coco_data["annotations"]),
            "avg_annotations_per_image": len(coco_data["annotations"]) / max(len(coco_data["images"]), 1),
            "category_distribution": {}
        }
        
        # Count per category
        for ann in coco_data["annotations"]:
            cat_id = ann["category_id"]
            cat_name = next(
                (c["name"] for c in coco_data["categories"] if c["id"] == cat_id),
                "unknown"
            )
            stats["category_distribution"][cat_name] = stats["category_distribution"].get(cat_name, 0) + 1
        
        # Save statistics
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\nDataset Statistics:")
        print(f"  Images: {stats['total_images']}")
        print(f"  Annotations: {stats['total_annotations']}")
        print(f"  Avg per image: {stats['avg_annotations_per_image']:.2f}")


def main():
    """Main entry point for pseudo-label generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for fashion images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--primary_threshold", type=float, default=0.35)
    parser.add_argument("--secondary_threshold", type=float, default=0.20)
    parser.add_argument("--ensemble_threshold", type=float, default=0.50)
    
    args = parser.parse_args()
    
    generator = PseudoLabelGenerator(
        primary_threshold=args.primary_threshold,
        secondary_threshold=args.secondary_threshold,
        ensemble_threshold=args.ensemble_threshold
    )
    
    generator.generate_dataset(
        args.input_dir,
        args.output_dir,
        args.max_images
    )


if __name__ == "__main__":
    main()
