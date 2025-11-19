from app.domain.repositories.ernie_repository import ErnieRepository

class AskErnieUseCase:
    """Use case for asking questions to the ERNIE model."""

    def __init__(self, repository: ErnieRepository):
        self._repository = repository

    def execute(self, image_url: str, prompt: str) -> str:
        """
        Execute the use case.

        Args:
            image_url: URL of the image.
            prompt: Question or prompt for the model.

        Returns:
            The model's response.
        """
        if not image_url:
            raise ValueError("Image URL cannot be empty")
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        return self._repository.generate(image_url=image_url, prompt=prompt)
