"""
Project VISTA: Zero-Shot Fashion Object Detection
================================================

A data-efficient framework for fashion object detection using vision-language models.

Modules:
    clip_classifier: CLIP-based zero-shot classification
    owl_detector: OWL-ViT open-vocabulary detection
    pseudo_labeler: Pseudo-label generation pipeline
    distillation: Knowledge distillation training
    evaluation: Detection evaluation metrics
    utils: Data loading and visualisation utilities
"""

from .clip_classifier import CLIPClassifier
from .owl_detector import OWLViTDetector, Detection, DetectionFilter
from .pseudo_labeler import PseudoLabelGenerator, PseudoLabel
from .distillation import DistillationTrainer, FashionDataset
from .evaluation import DetectionEvaluator, InferenceBenchmark

__version__ = "1.0.0"
__author__ = "Pablo Williams"

__all__ = [
    "CLIPClassifier",
    "OWLViTDetector",
    "Detection",
    "DetectionFilter",
    "PseudoLabelGenerator",
    "PseudoLabel",
    "DistillationTrainer",
    "FashionDataset",
    "DetectionEvaluator",
    "InferenceBenchmark"
]
