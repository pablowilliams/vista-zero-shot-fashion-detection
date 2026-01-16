"""
CLIP-based Zero-Shot Classification Module
Project VISTA: Zero-Shot Fashion Object Detection

This module implements zero-shot classification using CLIP embeddings
for fashion product category verification.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import clip
except ImportError:
    print("Installing CLIP...")
    import subprocess
    subprocess.check_call(["pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip


class CLIPClassifier:
    """
    Zero-shot classifier using CLIP for fashion category verification.
    
    CLIP (Contrastive Language-Image Pre-training) learns visual concepts
    from natural language supervision, enabling zero-shot transfer to
    downstream classification tasks without task-specific training.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialise CLIP classifier.
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14)
            device: Compute device (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Fashion-specific category prompts
        self.fashion_categories = [
            "a photo of a t-shirt",
            "a photo of a dress",
            "a photo of trousers",
            "a photo of a jacket",
            "a photo of a skirt",
            "a photo of a shirt",
            "a photo of shoes",
            "a photo of sneakers",
            "a photo of heels",
            "a photo of boots",
            "a photo of a handbag",
            "a photo of a backpack",
            "a photo of a watch",
            "a photo of sunglasses",
            "a photo of a belt",
            "a photo of a hat",
            "a photo of a scarf",
            "a photo of jewellery",
        ]
        
        # Pre-compute text embeddings for efficiency
        self._text_features = None
        self._precompute_text_features()
    
    def _precompute_text_features(self) -> None:
        """Pre-compute and cache text embeddings for fashion categories."""
        with torch.no_grad():
            text_tokens = clip.tokenize(self.fashion_categories).to(self.device)
            self._text_features = self.model.encode_text(text_tokens)
            self._text_features = self._text_features / self._text_features.norm(
                dim=-1, keepdim=True
            )
    
    def classify(
        self, 
        image: Image.Image, 
        candidate_labels: Optional[List[str]] = None,
        return_scores: bool = False
    ) -> Tuple[str, float]:
        """
        Perform zero-shot classification on an image.
        
        Args:
            image: PIL Image to classify
            candidate_labels: Custom labels to consider (uses defaults if None)
            return_scores: If True, return all category scores
            
        Returns:
            Tuple of (predicted_label, confidence_score)
            If return_scores=True, returns dict of all scores instead
        """
        # Use custom labels or pre-computed features
        if candidate_labels is not None:
            text_features = self._encode_custom_labels(candidate_labels)
            labels = candidate_labels
        else:
            text_features = self._text_features
            labels = self.fashion_categories
        
        # Encode image
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        scores = similarity[0].cpu().numpy()
        
        if return_scores:
            return {label: float(score) for label, score in zip(labels, scores)}
        
        best_idx = scores.argmax()
        return labels[best_idx], float(scores[best_idx])
    
    def _encode_custom_labels(self, labels: List[str]) -> torch.Tensor:
        """Encode custom text labels."""
        with torch.no_grad():
            # Add prompt template if not present
            prompted_labels = [
                f"a photo of {label}" if not label.startswith("a photo") else label
                for label in labels
            ]
            text_tokens = clip.tokenize(prompted_labels).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract CLIP embedding for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            512-dimensional (ViT-B) or 768-dimensional (ViT-L) embedding
        """
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()
    
    def compute_similarity(
        self, 
        image: Image.Image, 
        text: str
    ) -> float:
        """
        Compute cosine similarity between image and text.
        
        Args:
            image: PIL Image
            text: Text description
            
        Returns:
            Cosine similarity score in [0, 1]
        """
        with torch.no_grad():
            # Encode image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode text
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        return (similarity + 1) / 2  # Map from [-1, 1] to [0, 1]
    
    def batch_classify(
        self, 
        images: List[Image.Image], 
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Batch classification for efficiency.
        
        Args:
            images: List of PIL Images
            batch_size: Processing batch size
            
        Returns:
            List of (label, confidence) tuples
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            with torch.no_grad():
                # Stack preprocessed images
                image_tensors = torch.stack([
                    self.preprocess(img) for img in batch
                ]).to(self.device)
                
                # Encode batch
                image_features = self.model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                
                # Compute similarities
                similarities = (100.0 * image_features @ self._text_features.T)
                similarities = similarities.softmax(dim=-1)
                
                # Extract predictions
                for j in range(len(batch)):
                    scores = similarities[j].cpu().numpy()
                    best_idx = scores.argmax()
                    results.append((
                        self.fashion_categories[best_idx], 
                        float(scores[best_idx])
                    ))
        
        return results


def create_fashion_prompt_templates() -> Dict[str, List[str]]:
    """
    Create multiple prompt templates for robust zero-shot classification.
    
    Using diverse prompts improves classification robustness by averaging
    predictions across different textual formulations.
    """
    templates = {
        "simple": "a photo of {}",
        "fashion": "a fashion photo of {}",
        "product": "a product image of {}",
        "ecommerce": "an ecommerce listing for {}",
        "catalogue": "a catalogue image of {}",
        "clothing": "a piece of clothing: {}",
        "style": "a stylish {}",
    }
    return templates


if __name__ == "__main__":
    # Example usage
    classifier = CLIPClassifier()
    
    # Create a test image (solid colour for demonstration)
    test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    
    label, confidence = classifier.classify(test_image)
    print(f"Predicted: {label} (confidence: {confidence:.3f})")
    
    # Get all scores
    scores = classifier.classify(test_image, return_scores=True)
    print("\nAll category scores:")
    for cat, score in sorted(scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  {cat}: {score:.3f}")
