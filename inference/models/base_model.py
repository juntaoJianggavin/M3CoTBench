from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import sys

class BaseModel(ABC):
    """Base class for large models, defining common interfaces."""
    
    def __init__(
        self, 
        model_name: str,
        model_config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model.
            model_config: Configuration of the model.
            device: Device to load the model on.
        """
        self.model_name = model_name
        self.model_config = model_config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def eval(self):
        if hasattr(self, 'model') and hasattr(self.model, 'eval'):
            self.model.eval()
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(
        self, 
        image: Union[Image.Image, torch.Tensor],
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate text based on image and prompt.
        
        Args:
            image: Input image.
            prompt: Input prompt text.
            **kwargs: Additional arguments.
            
        Returns:
            Generated text.
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources, e.g., release GPU memory."""
        if hasattr(self, 'model'):
            # If the model is on CUDA, try to clear the cache
            if next(self.model.parameters(), torch.empty(0)).is_cuda:
                torch.cuda.empty_cache()
    
    def _prepare_inputs(
        self, 
        image: Union[Image.Image, torch.Tensor],
        prompt: str
    ) -> Dict[str, Any]:
        """
        Prepare inputs.
        
        Args:
            image: Input image.
            prompt: Input prompt text.
            
        Returns:
            Prepared input dictionary.
        """
        # This is a helper method that can be overridden in subclasses
        return {
            "image": image,
            "prompt": prompt
        }
    
    def _get_model_path(self) -> str:
        """
        Get model path.
        
        Returns:
            Model path.
        """
        return self.model_config.get("model_path", "")
    
    def _get_max_new_tokens(self) -> int:
        """
        Get maximum number of new tokens to generate.
        
        Returns:
            Maximum new tokens.
        """
        return self.model_config.get("max_new_tokens", 1024)
    
    def _get_temperature(self) -> float:
        """
        Get generation temperature.
        
        Returns:
            Temperature value.
        """
        return self.model_config.get("temperature", 0.1)