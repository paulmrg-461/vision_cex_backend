import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
from app.core.utils.logger import get_logger

_logger = get_logger("ernie_client")

class ErnieClient:
    """Client for interacting with the ERNIE-4.5-VL-28B-A3B-Thinking-AWQ-4bit model."""
    # ERNIE-4.5-VL-28B-A3B-Thinking-AWQ-4bit
    def __init__(self, model_name: str = "cyankiwi/ERNIE-4.5-VL-28B-A3B-Thinking-AWQ-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Loads the model and tokenizer."""
        try:
            _logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            _logger.info("Model loaded successfully.")
        except Exception as e:
            _logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_image(self, image_source: str) -> Image.Image:
        """Loads an image from a URL, base64 string, or local path."""
        try:
            if image_source.startswith("http://") or image_source.startswith("https://"):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            elif image_source.startswith("data:image"):
                # Base64 format: data:image/png;base64,iVBORw0KGgo...
                import base64
                header, encoded = image_source.split(",", 1)
                data = base64.b64decode(encoded)
                image = Image.open(BytesIO(data)).convert("RGB")
            else:
                # Assume local path
                import os
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"File not found: {image_source}")
                image = Image.open(image_source).convert("RGB")
            return image
        except Exception as e:
            _logger.error(f"Failed to load image from {image_source[:50]}...: {e}")
            raise ValueError(f"Failed to load image: {e}")

    def generate(self, image_url: str, prompt: str) -> str:
        """Generates a response for the given image and prompt."""
        if not self.model or not self.tokenizer:
            self._load_model()

        image = self._load_image(image_url)

        # Prepare messages
        messages = [{"role": "user", "content": prompt, "image": image}]
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response (simple heuristic, might need adjustment based on template)
        # The apply_chat_template usually handles the structure, so we might get the full conversation back.
        # We'll try to return the raw decoded text for now, or strip the prompt if possible.
        # A common pattern is that the output includes the prompt.
        if response.startswith(prompt_text):
             response = response[len(prompt_text):]
        
        return response.strip()
