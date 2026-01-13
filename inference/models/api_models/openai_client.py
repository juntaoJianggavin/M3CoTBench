# Filename: openai_client.py

from typing import Dict, Any, Optional, Union
import torch
from PIL import Image
import os
import base64
import io
import logging

# Import necessary exception types to propagate up
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError

from models.base_model import BaseModel

# Log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenAIModel(BaseModel):
    """
    Simplified OpenAI API model implementation.
    - Its sole responsibility is to perform a single API call.
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
        self.api_key = self.model_config.get("api_key", os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in model_config or OPENAI_API_KEY environment variable.")

        self.api_base = self.model_config.get("api_base", "https://api.openai.com/v1")
        self.openai_model = self.model_config.get("model", "gpt-4o")
        
        # Read timeout from config
        timeout = self.model_config.get("timeout", 120.0)
        
        # [Core Modification] Disable library-level automatic retries, leave retry logic entirely to InferenceEngine
        max_retries = 0 

        base_url = self.api_base.replace("/chat/completions", "")

        # Create synchronous client
        self.client = OpenAI(
            base_url=base_url, 
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        
        logging.info(f"API model [{self.openai_model}] loaded, API address: {base_url}")
        logging.info(f"Client configuration -> Timeout: {timeout}s, Internal retries disabled (controlled by engine)")

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

    def generate(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        prompt: str,
        **kwargs
    ) -> str:
        """
        [Synchronous] Generate text response for a single sample.
        If an API error occurs, this method will raise the exception directly to be handled by the external caller.
        """
        max_tokens = self.model_config.get("max_new_tokens", 1024)
        temperature = self.model_config.get("temperature", 0.1)
        
        base64_img = self._encode_image(image)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            {"type": "text", "text": prompt}
        ]}]

        # [Core Modification] Call API directly without catching exceptions. Let exceptions naturally bubble up to InferenceEngine.
        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        output = response.choices[0].message.content if response.choices else "API did not return choices"
        return output