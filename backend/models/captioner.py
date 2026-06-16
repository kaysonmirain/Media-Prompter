"""
Image Captioner using BLIP-2 (Salesforce) — state-of-the-art vision-language model.
Generates natural language descriptions of images.
"""
import re
import torch
from PIL import Image
from typing import Optional


def format_caption(text: str) -> str:
    """Normalize BLIP caption spacing, apostrophes, and capitalization."""
    if not text or not str(text).strip():
        return text

    t = re.sub(r"\s+", " ", str(text).strip())

    # Fix spaced apostrophes: water ' s -> water's, don ' t -> don't
    t = re.sub(r"(\w)\s+'\s*([a-z]{1,2})\b", r"\1'\2", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+'\s*([a-z]{1,2})\b", r"'\1", t, flags=re.IGNORECASE)

    # Remove stray spaces before punctuation
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)

    # Capitalize first letter
    if t:
        t = t[0].upper() + t[1:]

    # Capitalize first letter after sentence endings
    t = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        t,
    )

    return t.strip()


class ImageCaptioner:
    """
    BLIP-2 based image captioner. Generates rich, natural language
    descriptions of image content using a large vision-language model.
    Falls back to BLIP if BLIP-2 is unavailable or device is CPU only.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._load_model()

    def _load_model(self):
        from transformers import (
            BlipProcessor, BlipForConditionalGeneration,
            Blip2Processor, Blip2ForConditionalGeneration
        )

        # Temporarily force original BLIP for faster downloads and startup
        if False:
            try:
                print("[Captioner] Loading BLIP-2 (Salesforce/blip2-opt-2.7b)...")
                self.processor = Blip2Processor.from_pretrained(
                    "Salesforce/blip2-opt-2.7b"
                )
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.model.eval()
                self.model_name = "BLIP-2 (OPT-2.7B)"
                print("[Captioner] BLIP-2 loaded successfully")
                return
            except Exception as e:
                print(f"[Captioner] BLIP-2 unavailable ({e}), falling back to BLIP...")

        # Fallback to original BLIP (lighter, works on CPU)
        print("[Captioner] Loading BLIP (Salesforce/blip-image-captioning-large)...")
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        self.model.eval()
        self.model_name = "BLIP (Large)"
        print("[Captioner] BLIP loaded successfully")

    @torch.inference_mode()
    def caption(self, image: Image.Image, max_new_tokens: int = 150) -> dict:
        """
        Generate a natural language caption for the given PIL Image.
        Returns dict with caption text and model info.
        """
        image_rgb = image.convert("RGB")

        inputs = self.processor(
            images=image_rgb,
            return_tensors="pt"
        ).to(self.device)

        # For BLIP-2, use text prompt for conditional generation
        if "BLIP-2" in self.model_name:
            inputs["input_ids"] = self.processor.tokenizer(
                "Describe this image in detail:",
                return_tensors="pt"
            ).input_ids.to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            repetition_penalty=1.3,
            length_penalty=1.0
        )

        caption_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        # Clean up BLIP-2 prompt echo
        if caption_text.lower().startswith("describe this image"):
            caption_text = caption_text[caption_text.find(":") + 1:].strip()

        return {
            "text": format_caption(caption_text),
        }
