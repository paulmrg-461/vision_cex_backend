import logging
from dataclasses import dataclass
from typing import List, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceCaptionConfig:
    token: str
    model: str = "Salesforce/blip-image-captioning-base"
    base_url: str = "https://api-inference.huggingface.co"
    mode: str = "local"  # 'local' or 'api'


class HuggingFaceCaptionClient:
    """Client to generate image captions via Hugging Face.

    - local: uses transformers pipeline('image-to-text').
    - api: calls Hugging Face Inference API with image bytes.
    """

    def __init__(self, config: HuggingFaceCaptionConfig) -> None:
        self.config = config
        self._pipe = None
        self._local_available = False
        if self.config.mode.lower() == "local":
            try:
                import torch
                from transformers import pipeline  # type: ignore
                device = 0 if torch.cuda.is_available() else -1
                self._pipe = pipeline("image-to-text", model=self.config.model, device=device)
                self._local_available = True
                logger.info("Loaded local image-to-text pipeline: %s", self.config.model)
            except Exception as e:
                logger.error("Failed to init local captioning pipeline: %s", e)
                self._local_available = False

    def _sanitize_url(self, url: str) -> str:
        return url.strip().strip('`"')

    def describe(self, image_url: str, top_k: int = 1, max_new_tokens: int = 30) -> List[dict]:
        image_url = self._sanitize_url(image_url)
        if self.config.mode.lower() == "local" and self._local_available:
            return self._describe_local(image_url, top_k, max_new_tokens)
        return self._describe_api(image_url, top_k, max_new_tokens)

    def _describe_local(self, image_url: str, top_k: int, max_new_tokens: int) -> List[dict]:
        from PIL import Image
        from io import BytesIO
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        # num_return_sequences controls top_k candidates; ensure num_beams >= top_k
        num_beams = max(3, int(top_k))
        # Some versions of transformers' ImageToTextPipeline do not support
        # num_return_sequences/num_beams in _sanitize_parameters. To keep
        # compatibility, avoid passing those arguments and rely on the default
        # single caption output. The repository layer will slice to top_k if
        # multiple are produced by the pipeline.
        outputs = self._pipe(
            image,
            max_new_tokens=max_new_tokens,
        )
        # outputs typically: [{'generated_text': '...'}]
        return outputs

    def _describe_api(self, image_url: str, top_k: int, max_new_tokens: int) -> List[dict]:
        if not self.config.token:
            raise ValueError("HUGGINGFACE_TOKEN requerido para el modo API de captioning")
        # Fetch image bytes
        resp_img = requests.get(image_url, timeout=30)
        resp_img.raise_for_status()
        image_bytes = resp_img.content

        url = f"{self.config.base_url.rstrip('/')}/models/{self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
        }
        # Some captioning models accept parameters via query or JSON; Inference API for binary mode
        # allows sending bytes and optional 'X-Requested-With' headers or 'HF parameters' via query.
        # We'll keep it simple and rely on model defaults for now.
        resp = requests.post(url, headers=headers, data=image_bytes, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Respuesta no JSON del HuggingFace API: {resp.text[:200]}")
        # Expect list of dicts with 'generated_text'
        if isinstance(data, list):
            return data[:top_k]
        if isinstance(data, dict):
            # Some models may return a dict with 'generated_text' directly
            return [data]
        return []