"""
Knowledge Distillation Training Module
Project VISTA: Zero-Shot Fashion Object Detection

This module implements knowledge distillation from vision-language teacher
models to a lightweight YOLOv8 student detector.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# YOLOv8 from ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    from ultralytics import YOLO


class FashionDataset(Dataset):
    """
    Dataset for fashion object detection with pseudo-labels.
    """
    
    def __init__(
        self,
        image_dir: str,
        annotations_path: str,
        transform=None,
        img_size: int = 640
    ):
        """
        Initialise dataset.
        
        Args:
            image_dir: Directory containing images
            annotations_path: Path to COCO-format annotations JSON
            transform: Optional image transforms
            img_size: Target image size
        """
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Load annotations
        with open(annotations_path, "r") as f:
            self.coco_data = json.load(f)
        
        # Build image ID to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Build image ID to info mapping
        self.img_info = {img["id"]: img for img in self.coco_data["images"]}
        
        # List of image IDs
        self.image_ids = list(self.img_info.keys())
        
        # Category mapping
        self.num_classes = len(self.coco_data["categories"])
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get image and annotations.
        
        Returns:
            Tuple of (image_tensor, target_dict)
        """
        img_id = self.image_ids[idx]
        img_info = self.img_info[img_id]
        
        # Load image
        img_path = self.image_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        
        # Resize
        image = image.resize((self.img_size, self.img_size))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        # Convert to YOLO format (normalised xywh)
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            
            # Normalise coordinates
            x_center = (x + w / 2) / orig_width
            y_center = (y + h / 2) / orig_height
            w_norm = w / orig_width
            h_norm = h / orig_height
            
            boxes.append([x_center, y_center, w_norm, h_norm])
            labels.append(ann["category_id"] - 1)  # 0-indexed
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_id": img_id
        }
        
        return image_tensor, target


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation in object detection.
    
    Components:
    1. Hard loss: Standard detection loss on pseudo-labels
    2. Soft loss: KL divergence on softened logits from teacher
    3. Feature loss: L2 loss on intermediate features (optional)
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 4.0,
        feature_weight: float = 0.1
    ):
        """
        Initialise distillation loss.
        
        Args:
            alpha: Weight for soft loss (1 - alpha for hard loss)
            temperature: Softmax temperature for distillation
            feature_weight: Weight for feature alignment loss
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weight = feature_weight
        
        # Component losses
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_output: Dict,
        teacher_output: Optional[Dict],
        targets: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined distillation loss.
        
        Args:
            student_output: Student model predictions
            teacher_output: Teacher model predictions (optional)
            targets: Ground truth targets
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        losses = {}
        
        # Hard loss (student vs pseudo-labels)
        hard_bbox = self.bbox_loss(
            student_output["boxes"],
            targets["boxes"]
        )
        hard_cls = self.cls_loss(
            student_output["logits"],
            targets["labels"]
        )
        hard_loss = hard_bbox + hard_cls
        losses["hard_loss"] = hard_loss.item()
        
        # Soft loss (student vs teacher)
        if teacher_output is not None:
            # Softened class probabilities
            soft_student = F.log_softmax(
                student_output["logits"] / self.temperature,
                dim=-1
            )
            soft_teacher = F.softmax(
                teacher_output["logits"] / self.temperature,
                dim=-1
            )
            soft_cls = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
            soft_cls = soft_cls * (self.temperature ** 2)
            
            # Box regression distillation
            soft_bbox = self.bbox_loss(
                student_output["boxes"],
                teacher_output["boxes"]
            )
            
            soft_loss = soft_cls + soft_bbox
            losses["soft_loss"] = soft_loss.item()
            
            # Feature alignment (if available)
            if "features" in student_output and "features" in teacher_output:
                feature_loss = F.mse_loss(
                    student_output["features"],
                    teacher_output["features"]
                )
                losses["feature_loss"] = feature_loss.item()
            else:
                feature_loss = 0
            
            total_loss = (
                (1 - self.alpha) * hard_loss +
                self.alpha * soft_loss +
                self.feature_weight * feature_loss
            )
        else:
            total_loss = hard_loss
        
        losses["total_loss"] = total_loss.item()
        
        return total_loss, losses


class DistillationTrainer:
    """
    Trainer for knowledge distillation from VLM teacher to YOLO student.
    """
    
    def __init__(
        self,
        student_model: str = "yolov8n.pt",
        num_classes: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialise trainer.
        
        Args:
            student_model: Path to pretrained YOLO model or model name
            num_classes: Number of object classes
            device: Compute device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Load student model
        print(f"Loading student model: {student_model}")
        self.student = YOLO(student_model)
        
        # Loss function
        self.criterion = DistillationLoss()
    
    def train(
        self,
        train_dataset: FashionDataset,
        val_dataset: Optional[FashionDataset] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        output_dir: str = "models"
    ) -> Dict:
        """
        Train student model on pseudo-labels.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            output_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # For YOLO, we use the built-in training API
        # First, convert pseudo-labels to YOLO format
        yolo_data_dir = self._prepare_yolo_dataset(train_dataset, val_dataset)
        
        # Train using YOLO API
        results = self.student.train(
            data=str(yolo_data_dir / "data.yaml"),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            lr0=learning_rate,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            project=str(output_dir),
            name="vista_student",
            exist_ok=True,
            verbose=True
        )
        
        return results
    
    def _prepare_yolo_dataset(
        self,
        train_dataset: FashionDataset,
        val_dataset: Optional[FashionDataset]
    ) -> Path:
        """
        Convert COCO format to YOLO format.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Path to YOLO dataset directory
        """
        yolo_dir = Path("data/yolo_format")
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        
        if val_dataset:
            (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Convert training data
        self._convert_to_yolo(train_dataset, yolo_dir, "train")
        
        if val_dataset:
            self._convert_to_yolo(val_dataset, yolo_dir, "val")
        
        # Create data.yaml
        categories = train_dataset.coco_data["categories"]
        class_names = [cat["name"] for cat in sorted(categories, key=lambda x: x["id"])]
        
        data_yaml = f"""
path: {yolo_dir.absolute()}
train: images/train
val: images/{"val" if val_dataset else "train"}

nc: {len(class_names)}
names: {class_names}
"""
        
        with open(yolo_dir / "data.yaml", "w") as f:
            f.write(data_yaml)
        
        return yolo_dir
    
    def _convert_to_yolo(
        self,
        dataset: FashionDataset,
        output_dir: Path,
        split: str
    ) -> None:
        """
        Convert dataset to YOLO format.
        
        Args:
            dataset: FashionDataset instance
            output_dir: Output directory
            split: 'train' or 'val'
        """
        import shutil
        
        for img_id in tqdm(dataset.image_ids, desc=f"Converting {split} data"):
            img_info = dataset.img_info[img_id]
            
            # Copy image
            src_path = dataset.image_dir / img_info["file_name"]
            dst_path = output_dir / "images" / split / img_info["file_name"]
            
            if src_path.exists():
                shutil.copy(src_path, dst_path)
            
            # Write label file
            anns = dataset.img_to_anns.get(img_id, [])
            label_path = output_dir / "labels" / split / (img_info["file_name"].rsplit(".", 1)[0] + ".txt")
            
            with open(label_path, "w") as f:
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    orig_w = img_info["width"]
                    orig_h = img_info["height"]
                    
                    # Convert to YOLO format (class x_center y_center width height)
                    x_center = (x + w / 2) / orig_w
                    y_center = (y + h / 2) / orig_h
                    w_norm = w / orig_w
                    h_norm = h / orig_h
                    
                    class_id = ann["category_id"] - 1  # 0-indexed
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    def export(self, output_path: str, format: str = "torchscript") -> None:
        """
        Export trained model.
        
        Args:
            output_path: Output file path
            format: Export format (torchscript, onnx, etc.)
        """
        self.student.export(format=format)


def train_from_pseudo_labels(
    image_dir: str,
    annotations_path: str,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 32
) -> None:
    """
    Convenience function to train student model from pseudo-labels.
    
    Args:
        image_dir: Directory containing images
        annotations_path: Path to COCO-format pseudo-labels
        output_dir: Output directory for model
        epochs: Training epochs
        batch_size: Batch size
    """
    # Create dataset
    dataset = FashionDataset(image_dir, annotations_path)
    
    # Create trainer
    trainer = DistillationTrainer(num_classes=dataset.num_classes)
    
    # Train
    trainer.train(
        train_dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )


def main():
    """Main entry point for distillation training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train student model via knowledge distillation")
    parser.add_argument("--pseudo_labels", type=str, required=True, help="Path to pseudo-labels JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    train_from_pseudo_labels(
        image_dir=args.image_dir,
        annotations_path=args.pseudo_labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
