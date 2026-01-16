"""
Utility Functions
Project VISTA: Zero-Shot Fashion Object Detection

This module provides data loading, visualisation, and helper utilities.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


# Colour palette for visualisation
CATEGORY_COLOURS = {
    "topwear": "#FF6B6B",
    "bottomwear": "#4ECDC4",
    "dress": "#9B59B6",
    "footwear": "#F39C12",
    "bag": "#2ECC71",
    "accessories": "#3498DB",
    "default": "#95A5A6"
}


def load_fashion_dataset(
    image_dir: str,
    metadata_path: str
) -> Dict:
    """
    Load Fashion Product Images dataset.
    
    Args:
        image_dir: Directory containing images
        metadata_path: Path to styles.csv metadata
        
    Returns:
        Dictionary with images and metadata
    """
    import pandas as pd
    
    # Load metadata
    df = pd.read_csv(metadata_path, on_error="ignore")
    
    # Build image to metadata mapping
    data = {
        "images": [],
        "metadata": {}
    }
    
    for _, row in df.iterrows():
        img_id = str(row["id"])
        img_path = Path(image_dir) / f"{img_id}.jpg"
        
        if img_path.exists():
            data["images"].append(str(img_path))
            data["metadata"][img_id] = {
                "gender": row.get("gender", "Unknown"),
                "masterCategory": row.get("masterCategory", "Unknown"),
                "subCategory": row.get("subCategory", "Unknown"),
                "articleType": row.get("articleType", "Unknown"),
                "baseColour": row.get("baseColour", "Unknown"),
                "season": row.get("season", "Unknown"),
                "usage": row.get("usage", "Unknown"),
                "productDisplayName": row.get("productDisplayName", "Unknown")
            }
    
    return data


def visualise_detections(
    image: Union[Image.Image, str, np.ndarray],
    detections: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_width: int = 2,
    font_size: int = 12
) -> Image.Image:
    """
    Visualise object detections on an image.
    
    Args:
        image: PIL Image, path, or numpy array
        detections: List of detection dictionaries with bbox, label, confidence
        show_labels: Whether to display labels
        show_confidence: Whether to display confidence scores
        line_width: Bounding box line width
        font_size: Label font size
        
    Returns:
        PIL Image with visualised detections
    """
    # Load image if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.copy()
    
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        bbox = det.get("bbox", det.get("box", []))
        label = det.get("label", det.get("category", "unknown"))
        confidence = det.get("confidence", det.get("score", 0))
        
        # Get colour
        colour = CATEGORY_COLOURS.get(label.lower(), CATEGORY_COLOURS["default"])
        
        # Draw bounding box
        if len(bbox) == 4:
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                # Format: [x1, y1, x2, y2]
                draw.rectangle(bbox, outline=colour, width=line_width)
            else:
                # Format: [x, y, w, h]
                x, y, w, h = bbox
                draw.rectangle([x, y, x + w, y + h], outline=colour, width=line_width)
        
        # Draw label
        if show_labels:
            text = label
            if show_confidence:
                text += f" {confidence:.2f}"
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position above the box
            text_x = bbox[0]
            text_y = max(0, bbox[1] - text_height - 4)
            
            # Draw background rectangle
            draw.rectangle(
                [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                fill=colour
            )
            
            # Draw text
            draw.text((text_x + 2, text_y + 2), text, fill="white", font=font)
    
    return image


def create_detection_mosaic(
    images: List[Union[Image.Image, str]],
    detections_list: List[List[Dict]],
    grid_size: Tuple[int, int] = (3, 3),
    image_size: int = 256
) -> Image.Image:
    """
    Create a mosaic of images with detections.
    
    Args:
        images: List of images
        detections_list: List of detection lists (one per image)
        grid_size: (rows, cols) for mosaic
        image_size: Size of each image in mosaic
        
    Returns:
        Mosaic PIL Image
    """
    rows, cols = grid_size
    num_images = min(len(images), rows * cols)
    
    mosaic = Image.new("RGB", (cols * image_size, rows * image_size), color="white")
    
    for i in range(num_images):
        # Load and resize image
        img = images[i]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        img = img.resize((image_size, image_size))
        
        # Scale detections
        if i < len(detections_list):
            orig_size = images[i].size if isinstance(images[i], Image.Image) else Image.open(images[i]).size
            scale_x = image_size / orig_size[0]
            scale_y = image_size / orig_size[1]
            
            scaled_dets = []
            for det in detections_list[i]:
                bbox = det["bbox"]
                scaled_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ]
                scaled_dets.append({**det, "bbox": scaled_bbox})
            
            img = visualise_detections(img, scaled_dets, font_size=10)
        
        # Place in mosaic
        row = i // cols
        col = i % cols
        mosaic.paste(img, (col * image_size, row * image_size))
    
    return mosaic


def save_coco_annotations(
    annotations: List[Dict],
    images: List[Dict],
    categories: List[Dict],
    output_path: str
) -> None:
    """
    Save annotations in COCO format.
    
    Args:
        annotations: List of annotation dictionaries
        images: List of image info dictionaries
        categories: List of category dictionaries
        output_path: Output JSON path
    """
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {
            "description": "Project VISTA Fashion Detection Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "Pablo Williams"
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)


def compute_dataset_statistics(annotations_path: str) -> Dict:
    """
    Compute statistics for a COCO-format dataset.
    
    Args:
        annotations_path: Path to COCO annotations JSON
        
    Returns:
        Statistics dictionary
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)
    
    stats = {
        "num_images": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "num_categories": len(data["categories"]),
        "avg_annotations_per_image": len(data["annotations"]) / max(len(data["images"]), 1),
        "category_distribution": defaultdict(int),
        "bbox_statistics": {
            "widths": [],
            "heights": [],
            "areas": [],
            "aspect_ratios": []
        }
    }
    
    # Category distribution
    cat_names = {cat["id"]: cat["name"] for cat in data["categories"]}
    for ann in data["annotations"]:
        cat_name = cat_names.get(ann["category_id"], "unknown")
        stats["category_distribution"][cat_name] += 1
        
        # Bbox statistics
        x, y, w, h = ann["bbox"]
        stats["bbox_statistics"]["widths"].append(w)
        stats["bbox_statistics"]["heights"].append(h)
        stats["bbox_statistics"]["areas"].append(w * h)
        stats["bbox_statistics"]["aspect_ratios"].append(w / max(h, 1))
    
    # Convert to summary statistics
    for key in ["widths", "heights", "areas", "aspect_ratios"]:
        values = stats["bbox_statistics"][key]
        if values:
            stats["bbox_statistics"][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
    
    stats["category_distribution"] = dict(stats["category_distribution"])
    
    return stats


def plot_training_curves(
    history: Dict,
    output_path: Optional[str] = None
) -> None:
    """
    Plot training curves.
    
    Args:
        history: Dictionary with training metrics per epoch
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    if "loss" in history:
        axes[0, 0].plot(history["loss"], label="Training Loss")
        if "val_loss" in history:
            axes[0, 0].plot(history["val_loss"], label="Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # mAP curve
    if "mAP" in history:
        axes[0, 1].plot(history["mAP"], label="mAP@0.5")
        if "mAP_95" in history:
            axes[0, 1].plot(history["mAP_95"], label="mAP@0.5:0.95")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mAP")
        axes[0, 1].set_title("mAP Curves")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Precision/Recall
    if "precision" in history and "recall" in history:
        axes[1, 0].plot(history["precision"], label="Precision")
        axes[1, 0].plot(history["recall"], label="Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("Precision and Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if "lr" in history:
        axes[1, 1].plot(history["lr"])
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def split_dataset(
    annotations_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Split COCO dataset into train/val/test.
    
    Args:
        annotations_path: Path to full annotations
        output_dir: Output directory for split files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    np.random.seed(seed)
    
    with open(annotations_path, "r") as f:
        data = json.load(f)
    
    # Shuffle image IDs
    image_ids = [img["id"] for img in data["images"]]
    np.random.shuffle(image_ids)
    
    # Split
    n = len(image_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        "train": set(image_ids[:train_end]),
        "val": set(image_ids[train_end:val_end]),
        "test": set(image_ids[val_end:])
    }
    
    # Create split files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_ids in splits.items():
        split_images = [img for img in data["images"] if img["id"] in split_ids]
        split_anns = [ann for ann in data["annotations"] if ann["image_id"] in split_ids]
        
        split_data = {
            "images": split_images,
            "annotations": split_anns,
            "categories": data["categories"]
        }
        
        with open(output_dir / f"{split_name}.json", "w") as f:
            json.dump(split_data, f, indent=2)
        
        print(f"{split_name}: {len(split_images)} images, {len(split_anns)} annotations")


if __name__ == "__main__":
    # Example usage
    print("Project VISTA Utilities")
    print("=" * 40)
    
    # Create sample visualisation
    img = Image.new("RGB", (640, 480), color=(200, 200, 200))
    sample_detections = [
        {"bbox": [100, 100, 300, 400], "label": "dress", "confidence": 0.92},
        {"bbox": [350, 150, 550, 350], "label": "bag", "confidence": 0.87}
    ]
    
    vis_img = visualise_detections(img, sample_detections)
    print("Sample visualisation created successfully")
