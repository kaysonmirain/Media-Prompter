"""
Media Prompter — Deep Vision Analysis Engine
Core PyTorch-powered analyzer orchestrating multiple SOTA models.
"""
import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from PIL import Image
import numpy as np
from pathlib import Path
import json
import time
from typing import Optional, Callable

# ── Model sub-modules ───────────────────────────────────────────────────────
from models.detector import ObjectDetector
from models.captioner import ImageCaptioner
from models.classifier import CLIPAnalyzer


# ── Rich description database for common objects/scenes ─────────────────────
OBJECT_INFO = {
    "person": {
        "info": "Human subject detected. Facial landmarks, pose, and activity can be further analyzed.",
        "category": "Living Being"
    },
    "car": {
        "info": "Motor vehicle detected. Modern cars contain thousands of sensors and safety systems.",
        "category": "Vehicle"
    },
    "dog": {
        "info": "Domestic dog (Canis lupus familiaris). Over 340 recognized breeds exist worldwide.",
        "category": "Animal"
    },
    "cat": {
        "info": "Domestic cat (Felis catus). Cats have exceptional night vision and hearing beyond human range.",
        "category": "Animal"
    },
    "bicycle": {
        "info": "Human-powered two-wheeled vehicle. One of the most energy-efficient transport modes.",
        "category": "Vehicle"
    },
    "motorcycle": {
        "info": "Two-wheeled motor vehicle. Combines efficiency with speed for personal transport.",
        "category": "Vehicle"
    },
    "airplane": {
        "info": "Fixed-wing aircraft. Modern jets cruise at ~900 km/h at altitudes up to 43,000 ft.",
        "category": "Vehicle"
    },
    "bus": {
        "info": "Large road vehicle for passenger transport. Electric buses are rapidly becoming standard.",
        "category": "Vehicle"
    },
    "truck": {
        "info": "Heavy goods vehicle. Responsible for ~70% of freight delivery in most economies.",
        "category": "Vehicle"
    },
    "boat": {
        "info": "Watercraft detected. Vessels range from rowboats to massive container ships.",
        "category": "Vehicle"
    },
    "traffic light": {
        "info": "Traffic control device. Modern smart lights adapt timing based on traffic flow AI.",
        "category": "Infrastructure"
    },
    "fire hydrant": {
        "info": "Emergency water access point. Typically provides 250–500 gallons per minute.",
        "category": "Infrastructure"
    },
    "stop sign": {
        "info": "Regulatory road sign. The octagonal shape is universally recognized across 178 countries.",
        "category": "Infrastructure"
    },
    "bench": {
        "info": "Outdoor or indoor seating fixture. Found in parks, stations, and public spaces.",
        "category": "Furniture"
    },
    "bird": {
        "info": "Avian species detected. There are approximately 10,000 known bird species on Earth.",
        "category": "Animal"
    },
    "horse": {
        "info": "Equine mammal (Equus ferus caballus). Horses can sleep standing up.",
        "category": "Animal"
    },
    "sheep": {
        "info": "Domestic sheep (Ovis aries). Known for wool production; there are ~1 billion sheep globally.",
        "category": "Animal"
    },
    "cow": {
        "info": "Domestic cattle (Bos taurus). Their complex digestive system ferments cellulose.",
        "category": "Animal"
    },
    "elephant": {
        "info": "Largest land animal. Elephants have exceptional memory and complex social structures.",
        "category": "Animal"
    },
    "bear": {
        "info": "Large omnivorous mammal. Bears have excellent senses of smell — 7x stronger than dogs.",
        "category": "Animal"
    },
    "zebra": {
        "info": "African equid with distinctive black-and-white stripes. Each zebra's pattern is unique.",
        "category": "Animal"
    },
    "giraffe": {
        "info": "World's tallest living terrestrial animal. A giraffe's neck can be up to 1.8m long.",
        "category": "Animal"
    },
    "backpack": {
        "info": "Personal carry bag. Modern smart backpacks include charging ports and weight sensors.",
        "category": "Accessory"
    },
    "umbrella": {
        "info": "Rain/sun protection device. Invented in China over 3,000 years ago.",
        "category": "Accessory"
    },
    "handbag": {
        "info": "Personal bag or purse. A global fashion accessory with multi-billion dollar market.",
        "category": "Accessory"
    },
    "cell phone": {
        "info": "Mobile computing device. Modern smartphones contain more computing power than Apollo 11.",
        "category": "Electronics"
    },
    "laptop": {
        "info": "Portable personal computer. Modern laptops pack desktop-level performance in millimeters.",
        "category": "Electronics"
    },
    "tv": {
        "info": "Television display. 4K/8K OLED panels offer over 8 million pixels of resolution.",
        "category": "Electronics"
    },
    "keyboard": {
        "info": "Input device. QWERTY layout was designed in 1873 to slow typists and prevent jamming.",
        "category": "Electronics"
    },
    "mouse": {
        "info": "Pointing device. The first mouse was invented by Douglas Engelbart in 1964.",
        "category": "Electronics"
    },
    "remote": {
        "info": "Infrared or RF control device. Most use IR at 38kHz carrier frequency.",
        "category": "Electronics"
    },
    "refrigerator": {
        "info": "Food preservation appliance. Modern refrigerators maintain 2–4°C for optimal freshness.",
        "category": "Appliance"
    },
    "microwave": {
        "info": "Microwave oven. Uses 2.45 GHz electromagnetic radiation to excite water molecules.",
        "category": "Appliance"
    },
    "oven": {
        "info": "Cooking appliance. Convection ovens circulate air for 25% faster, more even cooking.",
        "category": "Appliance"
    },
    "sink": {
        "info": "Plumbing fixture. Touchless sensor faucets reduce water usage by up to 70%.",
        "category": "Fixture"
    },
    "toilet": {
        "info": "Sanitation fixture. Modern dual-flush toilets save thousands of liters annually.",
        "category": "Fixture"
    },
    "chair": {
        "info": "Seating furniture. Ergonomic chairs are designed using biomechanical research.",
        "category": "Furniture"
    },
    "couch": {
        "info": "Upholstered seating furniture. Also called sofa or settee depending on region.",
        "category": "Furniture"
    },
    "bed": {
        "info": "Sleep furniture. Adults spend ~26 years sleeping in their lifetime.",
        "category": "Furniture"
    },
    "dining table": {
        "info": "Meal surface furniture. Historically, dining tables were status symbols of wealth.",
        "category": "Furniture"
    },
    "book": {
        "info": "Printed or bound knowledge medium. Over 130 million unique books have been published.",
        "category": "Object"
    },
    "clock": {
        "info": "Timekeeping device. Atomic clocks are accurate to within 1 second per 300 million years.",
        "category": "Object"
    },
    "vase": {
        "info": "Decorative or functional vessel. Ceramic vases date back over 18,000 years.",
        "category": "Object"
    },
    "scissors": {
        "info": "Cutting tool. The pivot-blade design has remained fundamentally unchanged for 2,000 years.",
        "category": "Tool"
    },
    "bottle": {
        "info": "Liquid container. Recycling one glass bottle saves enough energy to power a PC for 30 min.",
        "category": "Object"
    },
    "cup": {
        "info": "Drinking vessel. Humans have used cup-like vessels for at least 7,500 years.",
        "category": "Object"
    },
    "bowl": {
        "info": "Open container for food or liquids. The shape maximizes capacity with minimal material.",
        "category": "Object"
    },
    "knife": {
        "info": "Cutting utensil. One of humanity's oldest tools, dating back 2.5 million years.",
        "category": "Tool"
    },
    "fork": {
        "info": "Pronged eating utensil. Forks became common in Europe only after the 11th century.",
        "category": "Object"
    },
    "spoon": {
        "info": "Scooping utensil. Among the oldest eating tools in human history.",
        "category": "Object"
    },
    "banana": {
        "info": "Tropical fruit (Musa spp). Bananas are slightly radioactive due to potassium-40.",
        "category": "Food"
    },
    "apple": {
        "info": "Pome fruit (Malus domestica). There are over 7,500 known apple cultivars worldwide.",
        "category": "Food"
    },
    "sandwich": {
        "info": "Filled bread dish. Named after the 4th Earl of Sandwich who popularized eating bread with filling.",
        "category": "Food"
    },
    "orange": {
        "info": "Citrus fruit. A single orange contains over 100% of the daily recommended vitamin C.",
        "category": "Food"
    },
    "broccoli": {
        "info": "Cruciferous vegetable. Contains sulforaphane, a powerful anti-cancer compound.",
        "category": "Food"
    },
    "carrot": {
        "info": "Root vegetable. Rich in beta-carotene, which converts to vitamin A in the body.",
        "category": "Food"
    },
    "hot dog": {
        "info": "Cured meat in a bun. Over 20 billion hot dogs are eaten in the US each year.",
        "category": "Food"
    },
    "pizza": {
        "info": "Italian flatbread dish. ~3 billion pizzas are consumed globally each year.",
        "category": "Food"
    },
    "donut": {
        "info": "Fried dough confection. The hole design ensures even cooking through the center.",
        "category": "Food"
    },
    "cake": {
        "info": "Sweet baked dessert. Ancient Egyptians were the first to bake cake-like bread with honey.",
        "category": "Food"
    },
    "potted plant": {
        "info": "Container-grown plant. Indoor plants can reduce CO₂ levels by up to 25%.",
        "category": "Plant"
    },
    "sports ball": {
        "info": "Athletic sphere. The specific sport can be determined by size, texture, and context.",
        "category": "Sport"
    },
    "kite": {
        "info": "Tethered flying device. Kites were invented in China around 470 BCE.",
        "category": "Toy/Sport"
    },
    "baseball bat": {
        "info": "Baseball striking implement. Professional bats are made from ash or maple wood.",
        "category": "Sport"
    },
    "skateboard": {
        "info": "Wheeled board for skating. First appeared in the late 1940s in California.",
        "category": "Sport"
    },
    "surfboard": {
        "info": "Wave-riding board. Modern performance boards use polyurethane foam and fiberglass.",
        "category": "Sport"
    },
    "tennis racket": {
        "info": "Strung sports implement. Modern graphite rackets can withstand over 200 lbs of string tension.",
        "category": "Sport"
    },
}


def get_object_info(label: str) -> dict:
    """Return enriched info for a detected object label."""
    label_lower = label.lower()
    if label_lower in OBJECT_INFO:
        return OBJECT_INFO[label_lower]
    return {
        "info": f"Detected object: '{label}'. This item was identified by the neural network with high confidence.",
        "category": "Unknown"
    }


class VisionAnalyzer:
    """
    Main orchestrator for all vision analysis tasks.
    Loads and manages all models, provides unified analyze() interface.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[Media Prompter] Initializing on device: {self.device}")
        self._load_models()

    def _load_models(self):
        """Load all models sequentially."""
        print("[Media Prompter] Loading Object Detector (YOLOv8x)...")
        self.detector = ObjectDetector(self.device)

        print("[Media Prompter] Loading Image Captioner (BLIP-2)...")
        self.captioner = ImageCaptioner(self.device)

        print("[Media Prompter] Loading CLIP Analyzer (ViT-L/14)...")
        self.clip_analyzer = CLIPAnalyzer(self.device)

        # EfficientNet-B7 scene classifier
        print("[Media Prompter] Loading Scene Classifier (EfficientNet-B7)...")
        weights = EfficientNet_B7_Weights.DEFAULT
        self.scene_model = efficientnet_b7(weights=weights).to(self.device)
        self.scene_model.eval()
        self.scene_transform = weights.transforms()
        self.imagenet_categories = weights.meta["categories"]

        print("[Media Prompter] All models loaded successfully.")

    @torch.inference_mode()
    def analyze_image(
        self,
        image: Image.Image,
        progress_cb: Optional[Callable] = None
    ) -> dict:
        """
        Full analysis pipeline for a single PIL Image.
        Returns a structured dict with all analysis results.
        """
        start_time = time.time()
        result = {}

        def _progress(step: str, pct: int):
            if progress_cb:
                progress_cb(step, pct)

        # ── 1. Object Detection ──────────────────────────────────────────────
        _progress("Detecting objects...", 10)
        detections = self.detector.detect(image)
        result["detections"] = detections

        # ── 2. Image Captioning ──────────────────────────────────────────────
        _progress("Generating caption...", 30)
        caption = self.captioner.caption(image)
        result["caption"] = caption

        # ── 3. CLIP Semantic Analysis ────────────────────────────────────────
        _progress("Running semantic analysis...", 50)
        clip_results = self.clip_analyzer.analyze(image)
        result["semantic_tags"] = clip_results["tags"]
        result["scene_type"] = clip_results["scene"]
        result["mood"] = clip_results["mood"]
        result["dominant_colors_approx"] = clip_results["colors"]

        # ── 4. Scene Classification (EfficientNet) ────────────────────────────
        _progress("Classifying scene...", 70)
        tensor = self.scene_transform(image).unsqueeze(0).to(self.device)
        logits = self.scene_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top5_probs, top5_idx = torch.topk(probs, 5)
        scene_classifications = [
            {
                "label": self.imagenet_categories[idx.item()],
                "confidence": round(prob.item() * 100, 2)
            }
            for idx, prob in zip(top5_idx, top5_probs)
        ]
        result["scene_classifications"] = scene_classifications

        # ── 5. Enrich detections with info ───────────────────────────────────
        _progress("Enriching results with knowledge database...", 85)
        for det in result["detections"]:
            info = get_object_info(det["label"])
            det.update(info)

        # ── 6. Summary ───────────────────────────────────────────────────────
        _progress("Generating analysis summary...", 95)
        object_counts = {}
        for det in result["detections"]:
            lbl = det["label"]
            object_counts[lbl] = object_counts.get(lbl, 0) + 1

        result["summary"] = {
            "total_objects_detected": len(result["detections"]),
            "unique_object_types": len(object_counts),
            "object_counts": object_counts,
            "top_scene": scene_classifications[0]["label"] if scene_classifications else "Unknown",
            "top_scene_confidence": scene_classifications[0]["confidence"] if scene_classifications else 0,
            "analysis_time_seconds": round(time.time() - start_time, 2),
        }

        _progress("Analysis complete!", 100)
        return result

    def analyze_image_from_path(self, path: str, progress_cb=None) -> dict:
        image = Image.open(path).convert("RGB")
        return self.analyze_image(image, progress_cb)
