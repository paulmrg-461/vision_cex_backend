from typing import List, Optional, Any, Dict

try:
    import requests  # type: ignore
    _requests_available = True
except Exception:
    requests = None  # type: ignore
    _requests_available = False

from app.core.utils.logger import get_logger

_logger = get_logger("deepseek_client")


class DeepSeekClient:
    """Cliente simple para la API de DeepSeek.

    Nota: Se asume un endpoint compatible con chat/completions y contenido multimodal.
    Si tu despliegue usa una variante distinta, ajusta `build_payload` y `endpoint_path`.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint_path = "/v1/chat/completions"  # ajustar si tu instancia requiere otra ruta

        if not _requests_available:
            _logger.error("'requests' no está disponible en el contenedor.")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_payload(self, images: List[str], locale: str = "es") -> Dict[str, Any]:
        """Construye el payload de solicitud para análisis de imágenes.

        Usa un formato estilo OpenAI/DeepSeek multimodal: un mensaje de usuario con partes de texto e imagen.
        """
        # Prompt de instrucciones para obtener JSON estricto con hallazgos de daños en buses.
        instruction = (
            "Eres un perito experto en evaluación de daños de buses. Analiza las fotos proporcionadas "
            "para identificar daños. Devuelve SOLO un JSON válido con los siguientes campos: \n"
            "{\n  \"summary\": string en idioma '{locale}',\n  \"findings\": [\n    {\n      \"part\": string,\n      \"type\": one of ['scratch','dent','crack','broken_light','broken_window','rust','paint_peel','deformation','missing_part'],\n      \"severity\": one of ['minor','moderate','major'],\n      \"confidence\": number between 0 and 1,\n      \"description\": string en idioma '{locale}',\n      \"bbox\": [x1,y1,x2,y2] opcional si es visible\n    }\n  ]\n}"
        ).format(locale=locale)

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": instruction}
        ]

        for img in images:
            # Formato estilo OpenAI: type=image_url con objeto {url: ...}
            content.append({"type": "image_url", "image_url": {"url": img}})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Eres un asistente experto en visión por computador y peritaje vehicular."},
                {"role": "user", "content": content},
            ],
            # Solicitar respuesta estrictamente en JSON si el proveedor lo soporta.
            "response_format": {"type": "json_object"}
        }

        return payload

    def analyze_images(self, images: List[str], locale: str = "es") -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY no configurada en entorno.")
        if not _requests_available:
            raise RuntimeError("La librería 'requests' no está disponible.")

        url = f"{self.base_url}{self.endpoint_path}"
        payload = self.build_payload(images, locale=locale)

        try:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=60)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                # La respuesta no es JSON; intentar extraer JSON de resp.text
                text = resp.text or ""
                _logger.warning("Respuesta DeepSeek no es JSON; intentando parsear texto.")
                import json, re
                # 1) intento directo
                try:
                    return json.loads(text)
                except Exception:
                    pass
                # 2) buscar fences
                fence = re.findall(r"```json\s*([\s\S]*?)```", text)
                if not fence:
                    fence = re.findall(r"```\s*([\s\S]*?)```", text)
                for block in fence:
                    try:
                        return json.loads(block)
                    except Exception:
                        continue
                # 3) objeto balanceado
                def extract_first_json_object(text_src: str):
                    stack = []
                    start = -1
                    for i, ch in enumerate(text_src):
                        if ch == '{':
                            if start == -1:
                                start = i
                            stack.append('{')
                        elif ch == '}':
                            if stack:
                                stack.pop()
                                if not stack and start != -1:
                                    snippet = text_src[start:i+1]
                                    try:
                                        return json.loads(snippet)
                                    except Exception:
                                        return None
                    return None
                obj = extract_first_json_object(text)
                if obj is not None:
                    return obj
                # sin parseo posible, devolver raw
                return {"raw_text": text}
        except Exception as e:
            _logger.error("Error llamando DeepSeek: %s", e)
            raise

        # Intentar extraer JSON del campo choices[0].message.content o equivalente
        # Intentar extraer JSON estructurado del contenido textual, soportando code fences y objetos anidados
        try:
            message = None
            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("delta")
                        if isinstance(msg, dict):
                            message = msg.get("content")
            if isinstance(message, str):
                import json
                # 1) Intento directo
                try:
                    return json.loads(message)
                except Exception:
                    pass

                # 2) Buscar bloque ```json ... ``` o ``` ... ```
                import re
                fence = re.findall(r"```json\s*([\s\S]*?)```", message)
                if not fence:
                    fence = re.findall(r"```\s*([\s\S]*?)```", message)
                for block in fence:
                    try:
                        return json.loads(block)
                    except Exception:
                        continue

                # 3) Extraer el primer objeto JSON balanceado
                def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
                    stack = []
                    start = -1
                    for i, ch in enumerate(text):
                        if ch == '{':
                            if start == -1:
                                start = i
                            stack.append('{')
                        elif ch == '}':
                            if stack:
                                stack.pop()
                                if not stack and start != -1:
                                    snippet = text[start:i+1]
                                    try:
                                        return json.loads(snippet)
                                    except Exception:
                                        return None
                    return None

                obj = extract_first_json_object(message)
                if obj is not None:
                    return obj
        except Exception:
            pass

        # Si no se pudo extraer JSON directo, devolver el objeto completo para que el repositorio lo interprete.
        return {"raw": data}