# Filename: claude_client.py
from typing import Dict, Any, Optional, Union
import torch
from PIL import Image
import os
import base64
import io
import logging

# Import necessary exception types
from anthropic import Anthropic, APIError, APITimeoutError, APIConnectionError, RateLimitError
from anthropic.types import Message

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClaudeModel(BaseModel):
    """
    Simplified Claude API model implementation.
    - Its sole responsibility is to make a single API call.
    - The external caller (InferenceEngine) is responsible for handling retry and error logic.
    """

    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        super().__init__(model_name, model_config, device)
        self.load()

    def load(self) -> None:
        """Load API configuration and create the client."""
        # Load API key from config file or environment variables
        self.api_key = (
            self.model_config.get("api_key") 
            or os.environ.get("CLAUDE_API_KEY") 
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Claude API key is required. Set it in model_config or as an environment variable.")

        self.api_base = self.model_config.get("api_base", "https://api.anthropic.com/v1")
        self.claude_model = self.model_config.get("model", "anthropic/claude-sonnet-4")
        
        timeout = self.model_config.get("timeout", 120.0)
        max_retries = 0

        # [Core Fix] Use auth_token parameter to pass the API key to adapt to api.aimlapi.com requirements
        self.client = Anthropic(
            base_url=self.api_base,
            auth_token=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        logging.info(f"Claude API model [{self.claude_model}] loaded, API address: {self.client.base_url}")
        logging.info(f"Client config -> Timeout: {timeout}s, Internal retries disabled (controlled by engine)")


    def _encode_image(self, image: Union[Image.Image, str, torch.Tensor]) -> str:
        """Unify image encoding to base64."""
        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4: image = image[0]
            image = image.detach().cpu().numpy()
            if image.shape[0] == 3: image = image.transpose(1, 2, 0)
            image = Image.fromarray((image * 255).astype('uint8'))
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _get_image_media_type(self, image: Union[Image.Image, str]) -> str:
        """Get media type based on file extension or default value."""
        if isinstance(image, str) and os.path.exists(image):
            ext = os.path.splitext(image)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                return "image/jpeg"
            elif ext == '.png':
                return "image/png"
            elif ext == '.gif':
                return "image/gif"
            elif ext == '.webp':
                return "image/webp"
        return "image/png"

    def generate(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        prompt: str,
        **kwargs
    ) -> str:
        """
        [Synchronous] Generate text response for a single sample (non-streaming).
        If an API error occurs, this method raises the exception directly to be handled by the external caller.
        """
        max_tokens = self.model_config.get("max_new_tokens", 4096)
        temperature = self.model_config.get("temperature", 0.1)
        system_prompt = kwargs.get("system", "You are a helpful AI assistant.")
        
        base64_img = self._encode_image(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": self._get_image_media_type(image), "data": base64_img}},
            {"type": "text", "text": prompt}
        ]}]

        # [Core Modification] Make a direct non-streaming call and let exceptions bubble up to InferenceEngine.
        response: Message = self.client.messages.create(
            model=self.claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            **kwargs
        )
        
        return response.content[0].text