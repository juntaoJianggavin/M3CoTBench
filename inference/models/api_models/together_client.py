# Filename: together_client.py

from typing import Dict, Any, Optional, Union
import torch
from PIL import Image
import os
import base64
import io
import logging

# Import Together library. If not installed, a prompt will be given in the load method.
# Together's exception types are compatible with OpenAI and can be reused directly for unified handling at higher levels.
try:
    from together import Together
    from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError
except ImportError:
    # If the library does not exist, define a placeholder first; a more specific error will be raised during actual use.
    Together = None
    APIError = APITimeoutError = APIConnectionError = RateLimitError = Exception

from models.base_model import BaseModel

# Log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TogetherModel(BaseModel):
    """
    Simplified Together API model implementation.
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
        self.client = None
        self.load()

    def load(self) -> None:
        """Load API configuration and create the client."""
        if Together is None:
            raise ImportError("The 'together' package is required to use TogetherModel. Please install it via 'pip install together'.")

        self.api_key = self.model_config.get("api_key", os.environ.get("TOGETHER_API_KEY"))
        if not self.api_key:
            raise ValueError("Together API key is required. Set it in model_config or TOGETHER_API_KEY environment variable.")

        # Base URL for Together API
        self.api_base = self.model_config.get("api_base", "https://api.together.xyz/v1")
        self.together_model = self.model_config.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        
        # Read timeout from config
        timeout = self.model_config.get("timeout", 120.0)
        
        # [Core Modification] Disable library-level automatic retries, leave retry logic entirely to InferenceEngine
        max_retries = 0 

        # Create synchronous client
        self.client = Together(
            base_url=self.api_base, 
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        
        logging.info(f"API model [{self.together_model}] loaded, API address: {self.api_base}")
        logging.info(f"Client configuration -> Timeout: {timeout}s, Internal retries disabled (controlled by engine)")

    def _encode_image(self, image: Union[Image.Image, str, torch.Tensor]) -> str:
        """
        Unify image encoding to base64.
        This method is identical to the implementation in openai_client, used to handle different formats of image input.
        """
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
        image: Optional[Union[Image.Image, str, torch.Tensor]],
        prompt: str,
        **kwargs
    ) -> str:
        """
        [Synchronous] Generate text response for a single sample.
        Supports pure text and image-text multimodal input.
        If an API error occurs, this method will raise the exception directly to be handled by the external caller.
        """
        max_tokens = self.model_config.get("max_new_tokens", 1024)
        # Models like Llama typically use slightly higher temperature
        temperature = self.model_config.get("temperature", 0.7)
        
        messages = []

        # 1. Add optional system prompt
        system_prompt = self.model_config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. Prepare user input content (text and optional image)
        user_content = []
        # If there is an image, encode it and add it to the content list
        if image:
            base64_img = self._encode_image(image)
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            )
        
        # Add text prompt
        user_content.append({"type": "text", "text": prompt})

        # 3. Build the final messages structure
        # OpenAI-compatible API requirement: content is a string for pure text, and a list for multimodal
        if len(user_content) == 1:
            # Text only
            messages.append({"role": "user", "content": user_content[0]["text"]})
        else:
            # Image and text mix
            messages.append({"role": "user", "content": user_content})

        # [Core Modification] Call API directly without catching exceptions. Let exceptions naturally bubble up to InferenceEngine.
        response = self.client.chat.completions.create(
            model=self.together_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        output = response.choices[0].message.content if response.choices else "API did not return choices"
        return output