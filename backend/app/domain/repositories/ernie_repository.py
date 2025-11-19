from abc import ABC, abstractmethod

class ErnieRepository(ABC):
    """Interface for interacting with the ERNIE model."""

    @abstractmethod
    def generate(self, image_url: str, prompt: str) -> str:
        """
        Generate a response from the ERNIE model based on an image and a prompt.

        Args:
            image_url: URL of the image to analyze.
            prompt: Text prompt to guide the generation.

        Returns:
            Generated text response.
        """
        raise NotImplementedError
