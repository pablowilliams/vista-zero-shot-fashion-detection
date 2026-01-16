"""
Evaluation Module
Project VISTA: Zero-Shot Fashion Object Detection

This module implements comprehensive evaluation metrics for object detection
including mAP, precision, recall, and inference benchmarking.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Installing pycocotools...")
    import subprocess
    subprocess.check_call(["pip", "install", "pycocotools"])
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval


class DetectionEvaluator:
    """
    Evaluates object detection performance using COCO metrics.
    
    Metrics computed:
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95: Mean AP averaged over IoU thresholds 0.5 to 0.95
    - Per-category precision, recall, and F1
    - Inference time benchmarks
    """
    
    def __init__(self, ground_truth_path: str):
        """
        Initialise evaluator with ground truth annotations.
        
        Args:
            ground_truth_path: Path to COCO-format ground truth JSON
        """
        self.coco_gt = COCO(ground_truth_path)
        self.categories = {
            cat["id"]: cat["name"] 
            for cat in self.coco_gt.loadCats(self.coco_gt.getCatIds())
        }
    
    def evaluate_predictions(
        self,
        predictions_path: str
    ) -> Dict:
        """
        Evaluate detection predictions against ground truth.
        
        Args:
            predictions_path: Path to COCO-format predictions JSON
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Load predictions
        coco_dt = self.coco_gt.loadRes(predictions_path)
        
        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        results = {
            "mAP@0.5": coco_eval.stats[1],  # AP at IoU=0.50
            "mAP@0.5:0.95": coco_eval.stats[0],  # AP at IoU=0.50:0.95
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "AR@1": coco_eval.stats[6],  # Average Recall with 1 detection
            "AR@10": coco_eval.stats[7],  # Average Recall with 10 detections
            "AR@100": coco_eval.stats[8],  # Average Recall with 100 detections
        }
        
        # Per-category evaluation
        results["per_category"] = self._evaluate_per_category(coco_eval)
        
        return results
    
    def _evaluate_per_category(self, coco_eval: COCOeval) -> Dict:
        """
        Compute per-category metrics.
        
        Args:
            coco_eval: COCOeval object after evaluation
            
        Returns:
            Dictionary of per-category metrics
        """
        per_category = {}
        
        precision = coco_eval.eval["precision"]
        # precision has shape [T, R, K, A, M]
        # T: IoU thresholds
        # R: recall thresholds
        # K: categories
        # A: area ranges
        # M: max detections
        
        for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
            cat_name = self.categories.get(cat_id, f"category_{cat_id}")
            
            # Get precision at IoU=0.5 (index 0)
            cat_precision = precision[0, :, cat_idx, 0, 2]  # IoU=0.5, all areas, max=100
            cat_precision = cat_precision[cat_precision > -1]
            
            if len(cat_precision) > 0:
                mean_precision = float(np.mean(cat_precision))
            else:
                mean_precision = 0.0
            
            per_category[cat_name] = {
                "AP@0.5": mean_precision,
                "num_gt": len(self.coco_gt.getAnnIds(catIds=[cat_id]))
            }
        
        return per_category
    
    def compute_confusion_matrix(
        self,
        predictions: List[Dict],
        iou_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Compute confusion matrix for predictions.
        
        Args:
            predictions: List of prediction dictionaries
            iou_threshold: IoU threshold for matching
            
        Returns:
            Confusion matrix as numpy array
        """
        num_classes = len(self.categories) + 1  # +1 for background
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        # Group predictions by image
        pred_by_image = defaultdict(list)
        for pred in predictions:
            pred_by_image[pred["image_id"]].append(pred)
        
        # Process each image
        for img_id in self.coco_gt.getImgIds():
            gt_anns = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=[img_id]))
            preds = pred_by_image.get(img_id, [])
            
            gt_matched = [False] * len(gt_anns)
            
            # Sort predictions by confidence
            preds = sorted(preds, key=lambda x: -x.get("score", 0))
            
            for pred in preds:
                pred_box = pred["bbox"]
                pred_cat = pred["category_id"]
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_ann in enumerate(gt_anns):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = self._compute_iou(pred_box, gt_ann["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt_cat = gt_anns[best_gt_idx]["category_id"]
                    confusion[gt_cat, pred_cat] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # False positive
                    confusion[0, pred_cat] += 1
            
            # Count false negatives
            for gt_idx, matched in enumerate(gt_matched):
                if not matched:
                    gt_cat = gt_anns[gt_idx]["category_id"]
                    confusion[gt_cat, 0] += 1
        
        return confusion
    
    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two boxes in COCO format [x, y, w, h].
        
        Args:
            box1: First box
            box2: Second box
            
        Returns:
            IoU value
        """
        # Convert to [x1, y1, x2, y2]
        b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        # Intersection
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-6)


class InferenceBenchmark:
    """
    Benchmarks model inference performance.
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Initialise benchmark.
        
        Args:
            model: Model to benchmark
            device: Compute device
        """
        self.model = model
        self.device = device
    
    def benchmark(
        self,
        image_dir: str,
        num_images: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Run inference benchmark.
        
        Args:
            image_dir: Directory containing test images
            num_images: Number of images to process
            warmup: Number of warmup iterations
            
        Returns:
            Benchmark results dictionary
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg"))[:num_images + warmup]
        
        if len(image_files) < warmup:
            raise ValueError(f"Need at least {warmup} images for warmup")
        
        # Warmup
        for img_path in image_files[:warmup]:
            image = Image.open(img_path).convert("RGB")
            _ = self.model(image)
        
        # Benchmark
        times = []
        for img_path in tqdm(image_files[warmup:], desc="Benchmarking"):
            image = Image.open(img_path).convert("RGB")
            
            start = time.perf_counter()
            _ = self.model(image)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
            "fps": 1000 / np.mean(times),
            "num_images": len(times)
        }


def compute_precision_recall_f1(
    predictions: List[Dict],
    ground_truth_path: str,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute precision, recall, and F1 per category.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth_path: Path to ground truth annotations
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with metrics per category
    """
    coco_gt = COCO(ground_truth_path)
    categories = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    
    # Count TP, FP, FN per category
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    # Group predictions by image
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["image_id"]].append(pred)
    
    # Process each image
    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        preds = pred_by_image.get(img_id, [])
        
        gt_matched = [False] * len(gt_anns)
        
        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: -x.get("score", 0))
        
        for pred in preds:
            pred_box = pred["bbox"]
            pred_cat = pred["category_id"]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_ann in enumerate(gt_anns):
                if gt_matched[gt_idx]:
                    continue
                if gt_ann["category_id"] != pred_cat:
                    continue
                
                iou = DetectionEvaluator._compute_iou(pred_box, gt_ann["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_cat] += 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_cat] += 1
        
        # Count false negatives
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                fn[gt_anns[gt_idx]["category_id"]] += 1
    
    # Compute metrics
    results = {}
    for cat_id, cat_name in categories.items():
        cat_tp = tp[cat_id]
        cat_fp = fp[cat_id]
        cat_fn = fn[cat_id]
        
        precision = cat_tp / max(cat_tp + cat_fp, 1)
        recall = cat_tp / max(cat_tp + cat_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        results[cat_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": cat_tp,
            "fp": cat_fp,
            "fn": cat_fn
        }
    
    # Compute weighted average
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    
    avg_precision = total_tp / max(total_tp + total_fp, 1)
    avg_recall = total_tp / max(total_tp + total_fn, 1)
    avg_f1 = 2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 1e-6)
    
    results["weighted_average"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }
    
    return results


def generate_evaluation_report(
    predictions_path: str,
    ground_truth_path: str,
    output_path: str
) -> None:
    """
    Generate comprehensive evaluation report.
    
    Args:
        predictions_path: Path to predictions JSON
        ground_truth_path: Path to ground truth JSON
        output_path: Output path for report
    """
    # Load predictions
    with open(predictions_path, "r") as f:
        predictions = json.load(f)
    
    # Run evaluation
    evaluator = DetectionEvaluator(ground_truth_path)
    coco_results = evaluator.evaluate_predictions(predictions_path)
    pr_results = compute_precision_recall_f1(predictions, ground_truth_path)
    
    # Compile report
    report = {
        "coco_metrics": coco_results,
        "precision_recall_f1": pr_results,
        "summary": {
            "mAP@0.5": coco_results["mAP@0.5"],
            "mAP@0.5:0.95": coco_results["mAP@0.5:0.95"],
            "weighted_precision": pr_results["weighted_average"]["precision"],
            "weighted_recall": pr_results["weighted_average"]["recall"],
            "weighted_f1": pr_results["weighted_average"]["f1"]
        }
    }
    
    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"mAP@0.5:        {coco_results['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95:   {coco_results['mAP@0.5:0.95']:.4f}")
    print(f"Precision:      {pr_results['weighted_average']['precision']:.4f}")
    print(f"Recall:         {pr_results['weighted_average']['recall']:.4f}")
    print(f"F1 Score:       {pr_results['weighted_average']['f1']:.4f}")
    print("=" * 60)


def main():
    """Main entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions JSON")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth JSON")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    generate_evaluation_report(
        args.predictions,
        args.ground_truth,
        args.output
    )


if __name__ == "__main__":
    main()
