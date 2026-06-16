"""
CLIP Analyzer using OpenCLIP ViT-L/14 — powerful zero-shot vision-language model.
Provides semantic tagging, scene classification, mood detection, and color analysis.
"""
import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Dict


# ── Zero-shot category lists ──────────────────────────────────────────────
SCENE_CATEGORIES = [
    "indoor scene", "outdoor scene", "urban landscape", "rural countryside",
    "forest", "beach", "mountain landscape", "desert", "ocean view",
    "city street", "building interior", "kitchen", "bedroom", "living room",
    "office workspace", "restaurant or cafe", "sports venue", "park or garden",
    "airport or station", "hospital or medical facility", "school or classroom",
    "shopping mall or store", "museum or gallery", "night scene", "daytime scene",
    "abstract or artistic", "underwater scene", "aerial view from above",
    "industrial or factory", "nature and wildlife"
]

MOOD_CATEGORIES = [
    "calm and peaceful", "exciting and energetic", "dark and mysterious",
    "bright and cheerful", "sad and melancholic", "romantic and intimate",
    "dramatic and intense", "playful and fun", "professional and formal",
    "natural and organic", "futuristic and technological", "vintage and nostalgic",
    "scary and unsettling", "cozy and warm", "cold and isolated"
]

COLOR_CATEGORIES = [
    "predominantly red tones", "predominantly blue tones", "predominantly green tones",
    "predominantly yellow and orange tones", "predominantly purple and violet tones",
    "predominantly white and light tones", "predominantly black and dark tones",
    "warm color palette", "cool color palette", "neutral gray tones",
    "vibrant and colorful", "monochromatic", "pastel colors"
]

SEMANTIC_TAGS = [
    "technology", "nature", "animals", "people", "food and drink",
    "sports", "art and culture", "transportation", "architecture",
    "fashion and clothing", "health and medicine", "education",
    "entertainment", "business and work", "travel and tourism",
    "science and research", "music", "gaming", "environment",
    "weather and sky", "water and ocean", "plants and vegetation",
    "night life", "children and family", "elderly", "celebration",
    "danger and emergency", "luxury and wealth", "poverty and hardship",
    "diversity and inclusion", "historical", "modern and contemporary"
]


class CLIPAnalyzer:
    """
    OpenCLIP ViT-L/14 zero-shot classification for semantic understanding.
    Provides tags, scene type, mood, and dominant color palette analysis.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._load_model()
        self._precompute_embeddings()

    def _load_model(self):
        print("[CLIP] Loading OpenCLIP ViT-L/14 (openai)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[CLIP] OpenCLIP ViT-L/14 loaded")

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text prompts into normalized CLIP embeddings."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _precompute_embeddings(self):
        """Pre-compute text embeddings for all category lists for speed."""
        print("[CLIP] Pre-computing text embeddings...")
        self.scene_emb = self._encode_texts(SCENE_CATEGORIES)
        self.mood_emb = self._encode_texts(MOOD_CATEGORIES)
        self.color_emb = self._encode_texts(COLOR_CATEGORIES)
        self.tag_emb = self._encode_texts(SEMANTIC_TAGS)
        print("[CLIP] Text embeddings ready")

    @torch.inference_mode()
    def _classify(
        self,
        img_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        categories: List[str],
        top_k: int = 3
    ) -> List[Dict]:
        """Classify image features against pre-computed text embeddings."""
        similarity = (img_features @ text_embeddings.T).squeeze(0)
        probs = torch.softmax(similarity * 100, dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, min(top_k, len(categories)))

        return [
            {
                "label": categories[idx.item()],
                "confidence": round(prob.item() * 100, 1)
            }
            for idx, prob in zip(top_k_idx, top_k_probs)
        ]

    @torch.inference_mode()
    def analyze(self, image: Image.Image) -> Dict:
        """
        Full CLIP analysis: scene, mood, colors, semantic tags.
        Returns structured dict with all results.
        """
        image_rgb = image.convert("RGB")
        img_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)

        # Encode image
        img_features = self.model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Classify across all dimensions
        scenes = self._classify(img_features, self.scene_emb, SCENE_CATEGORIES, top_k=3)
        moods = self._classify(img_features, self.mood_emb, MOOD_CATEGORIES, top_k=3)
        colors = self._classify(img_features, self.color_emb, COLOR_CATEGORIES, top_k=3)
        tags = self._classify(img_features, self.tag_emb, SEMANTIC_TAGS, top_k=6)

        return {
            "scene": scenes[0]["label"] if scenes else "unknown",
            "scene_options": scenes,
            "mood": moods[0]["label"] if moods else "unknown",
            "mood_options": moods,
            "colors": colors,
            "tags": tags
        }
