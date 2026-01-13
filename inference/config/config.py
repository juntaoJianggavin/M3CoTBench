import os
import yaml
from typing import Dict, Any, List, Optional
import torch

class Config:
    """Configuration management class, responsible for loading and providing global configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the configuration file. Defaults to None, using models.yaml.
        """
        self.config_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.config_dir)
        
        # Default configuration file path
        if config_path is None:
            config_path = os.path.join(self.config_dir, "models.yaml")
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Set output directories
        self.output_dir = os.path.join(self.project_root, "output")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Model configuration dictionary.
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        return self.config["models"][model_name]
    
    def get_all_model_names(self) -> List[str]:
        """
        Get all configured model names.
        
        Returns:
            List of model names.
        """
        return list(self.config["models"].keys())
    
    def get_device(self) -> torch.device:
        """
        Get current device configuration.
        
        Returns:
            torch.device object.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_global_config(self) -> Dict[str, Any]:
        """
        Get global configuration.
        
        Returns:
            Global configuration dictionary.
        """
        return self.config.get("global", {})
    
    def get_device(self, gpu_ids=None):
        """
        Get device, supports manually specifying GPUs.
        
        Args:
            gpu_ids: List of GPU IDs or a comma-separated string.
            
        Returns:
            torch.device object.
        """
        from utils.device import setup_device
        return setup_device(gpu_ids)