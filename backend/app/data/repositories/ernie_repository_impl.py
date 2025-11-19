from app.domain.repositories.ernie_repository import ErnieRepository
from app.data.adapters.ernie_client import ErnieClient

class ErnieRepositoryImpl(ErnieRepository):
    """Implementation of ErnieRepository using ErnieClient."""

    def __init__(self, client: ErnieClient):
        self._client = client

    def generate(self, image_url: str, prompt: str) -> str:
        return self._client.generate(image_url=image_url, prompt=prompt)
