import json
import logging
import base64
from dataclasses import dataclass
from typing import Any, List, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceVqaConfig:
    token: str
    model: str = "dandelin/vilt-b32-finetuned-vqa"
    base_url: str = "https://api-inference.huggingface.co"
    mode: str = "api"  # 'api' or 'local'


class HuggingFaceVqaClient:
    """Client for Visual Question Answering using Hugging Face.

    Supports two modes:
    - 'api': calls the Hugging Face Inference API.
    - 'local': uses transformers ViltProcessor and ViltForQuestionAnswering locally.
    """

    def __init__(self, config: HuggingFaceVqaConfig) -> None:
        self.config = config
        self._local_available = False
        self._processor = None
        self._model = None

        if self.config.mode.lower() == "local":
            try:
                from transformers import ViltProcessor, ViltForQuestionAnswering  # type: ignore
                from PIL import Image  # noqa: F401
                self._processor = ViltProcessor.from_pretrained(self.config.model)
                self._model = ViltForQuestionAnswering.from_pretrained(self.config.model)
                self._local_available = True
                logger.info("Loaded local ViLT model for VQA: %s", self.config.model)
            except Exception as e:
                logger.error("Failed to initialize local VQA model. Falling back to API if available. Error: %s", e)
                self._local_available = False

    def _sanitize_url(self, url: str) -> str:
        return url.strip().strip('`"')

    def answer(self, question: str, image_url: str, top_k: int = 1) -> List[dict]:
        image_url = self._sanitize_url(image_url)
        try:
            print(f"[hf_vqa_client] mode={self.config.mode} model={self.config.model}")
        except Exception:
            pass
        if self.config.mode.lower() == "local" and self._local_available:
            try:
                print("[hf_vqa_client] using local VQA pipeline")
            except Exception:
                pass
            return self._answer_local(question, image_url, top_k)
        # Detect generative models (LLaVA / InstructBLIP) which require different payloads
        model_lc = self.config.model.lower()
        if any(k in model_lc for k in ("llava", "instructblip")):
            try:
                print(f"[hf_vqa_client] using generative API flow for model={self.config.model}")
            except Exception:
                pass
            return self._answer_api_generative(question, image_url)
        try:
            print(f"[hf_vqa_client] using classic API flow for model={self.config.model}")
        except Exception:
            pass
        return self._answer_api(question, image_url, top_k)

    def _answer_api(self, question: str, image_url: str, top_k: int) -> List[dict]:
        if not self.config.token:
            raise ValueError("HUGGINGFACE_TOKEN requerido para el modo API")
        url = f"{self.config.base_url.rstrip('/')}/models/{self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "inputs": {
                "question": question,
                "image": image_url,
            },
            "parameters": {
                "top_k": top_k,
            },
            "options": {"wait_for_model": True},
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Respuesta no JSON del HuggingFace API: {resp.text[:200]}")
        # API typically returns a list of answers with scores
        if isinstance(data, list):
            return data
        # Some endpoints may wrap under 'data' or similar
        if isinstance(data, dict):
            for key in ("data", "answers", "output"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        # Fallback to single answer
        return [data] if isinstance(data, dict) else []

    def _answer_api_generative(self, question: str, image_url: str) -> List[dict]:
        """Call HF Inference API for generative VQA models (e.g., LLaVA/InstructBLIP).

        Many generative vision-language models expect an image payload and a textual prompt,
        and typically return generated_text. This method sends the image as base64 along
        with the question as the prompt, and parses common response formats.
        """
        if not self.config.token:
            raise ValueError("HUGGINGFACE_TOKEN requerido para el modo API")

        # If using the HF Inference Router (OpenAI-compatible), call chat/completions
        base_lc = self.config.base_url.lower().rstrip('/')
        # Solo usar el flujo OpenAI-compatible si el base apunta al router raíz (no /hf-inference)
        use_router = ("router.huggingface.co" in base_lc) and (not base_lc.endswith("/hf-inference"))
        if use_router:
            # Usar la API OpenAI-compatible de Responses del Router HF
            endpoint = "https://router.huggingface.co/v1/responses"
            headers = {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            guidance = (
                "Responde en español en 4-6 frases. No contestes con 'sí' o 'no'. "
                "Describe piezas afectadas, deformaciones, roturas y severidad del daño."
            )
            # Construir input multimodal para Responses API
            input_blocks = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{question}\n\n{guidance}"},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
            payload = {
                "model": self.config.model,
                "input": input_blocks,
                "max_output_tokens": 512,
                "temperature": 0.2,
            }
            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code >= 400:
                raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text}")
            try:
                data = resp.json()
            except Exception:
                raise RuntimeError(f"Respuesta no JSON del HuggingFace API: {resp.text[:200]}")
            answers: List[dict] = []
            # Parse Responses API: output_text blocks
            if isinstance(data, dict) and "output" in data and isinstance(data["output"], list):
                for o in data["output"]:
                    if isinstance(o, dict) and o.get("type") == "output_text" and o.get("text"):
                        answers.append({"answer": str(o["text"]).strip(), "score": None})
            # Fallback a OpenAI-style choices si aplica
            if not answers and isinstance(data, dict) and "choices" in data:
                for ch in data.get("choices", []) or []:
                    msg = ch.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        answers.append({"answer": content.strip(), "score": None})
            if not answers:
                answers = [{"answer": json.dumps(data)[:200], "score": None}]
            return answers

        # Legacy Inference API path (model-specific endpoint)
        url = f"{self.config.base_url.rstrip('/')}/models/{self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        model_lc = self.config.model.lower()
        # Plantilla para fomentar respuestas largas en español
        guidance = (
            "Responde en español en 4-6 frases. No contestes con 'sí' o 'no'. "
            "Describe piezas afectadas, deformaciones, roturas y severidad del daño."
        )

        if model_lc.startswith("llava-hf/"):
            # Use chat-style messages as expected by the image-text-to-text pipeline
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_url": image_url},
                        {"type": "text", "text": f"{question}\n\n{guidance}"},
                    ],
                }
            ]
            payload = {
                "inputs": messages,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.2,
                },
                "options": {"wait_for_model": True},
            }
        else:
            # Default: send base64 image data URI with prompt
            resp_img = requests.get(image_url, timeout=30)
            resp_img.raise_for_status()
            img_bytes = resp_img.content
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{img_b64}"

            payload = {
                "inputs": {
                    "prompt": f"{question}\n\n{guidance}",
                    "image": data_uri,
                },
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.2,
                },
                "options": {"wait_for_model": True},
            }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Respuesta no JSON del HuggingFace API: {resp.text[:200]}")

        # Common formats:
        # - [{"generated_text": "..."}] (text-generation style)
        # - {"generated_text": "..."}
        # - {"choices": [{"text": "..."}]}
        answers: List[dict] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "generated_text" in item:
                    answers.append({"answer": str(item["generated_text"]), "score": None})
        elif isinstance(data, dict):
            if "generated_text" in data:
                answers.append({"answer": str(data["generated_text"]), "score": None})
            elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and ("text" in first or "generated_text" in first):
                    txt = first.get("text") or first.get("generated_text")
                    answers.append({"answer": str(txt), "score": None})

        # Fallback: stringify entire payload
        if not answers:
            answers = [{"answer": json.dumps(data)[:200], "score": None}]
        return answers

    def _answer_local(self, question: str, image_url: str, top_k: int) -> List[dict]:
        import requests
        from PIL import Image

        assert self._processor is not None and self._model is not None
        image = Image.open(requests.get(image_url, stream=True).raw)
        encoding = self._processor(image, question, return_tensors="pt")
        outputs = self._model(**encoding)
        logits = outputs.logits
        probs = logits.softmax(-1)[0]
        # Get top-k indices
        topk_vals, topk_indices = probs.topk(top_k)
        answers = []
        for score, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
            label = self._model.config.id2label.get(idx, str(idx))
            answers.append({"answer": label, "score": float(score)})
        return answers