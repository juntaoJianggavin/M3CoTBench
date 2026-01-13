from typing import Dict, Any, List, Optional
import torch
import importlib
import inspect
import os

from models.base_model import BaseModel

def get_model(
    model_name: str, 
    model_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> BaseModel:
    """
    Get model instance by name.
    
    Args:
        model_name: Model name
        model_config: Model configuration
        device: Device
        
    Returns:
        Model instance
    """
    model_type = model_config.get("type", "general")
    module_name = model_config.get("module", "")
    
    if not module_name:
        raise ValueError(f"Module not specified for model {model_name}")
    
    # Determine module path based on model type
    if model_type == "general":
        package_path = "models.general_models"
    elif model_type == "medical":
        package_path = "models.medical_models"
    elif model_type == "api":
        package_path = "models.api_models"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Import module
    try:
        module = importlib.import_module(f"{package_path}.{module_name}")
    except ImportError as e:
        raise ImportError(f"Failed to import module {package_path}.{module_name}: {str(e)}")
    
    # Find model class
    # Convention: The module should contain a class named XxxModel, where Xxx is the PascalCase form of the module name
    model_class_name = ''.join(word.capitalize() for word in module_name.split('_')) + 'Model'
    
    if hasattr(module, model_class_name):
        model_class = getattr(module, model_class_name)
        if inspect.isclass(model_class) and issubclass(model_class, BaseModel):
            return model_class(model_name, model_config, device)
        else:
            raise TypeError(f"Found {model_class_name} in {module_name}, but it is not a subclass of BaseModel")
    else:
        # Attempt to find any model class
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj != BaseModel:
                return obj(model_name, model_config, device)
        
        raise ValueError(f"No model class found in module {module_name}")

def list_available_models() -> List[str]:
    """
    List all available models.
    
    Returns:
        List of model names
    """
    models = []
    
    # Iterate through general models directory
    general_dir = os.path.join(os.path.dirname(__file__), "general_models")
    if os.path.exists(general_dir):
        for filename in os.listdir(general_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                models.append(f"general/{filename[:-3]}")
    
    # Iterate through medical models directory
    medical_dir = os.path.join(os.path.dirname(__file__), "medical_models")
    if os.path.exists(medical_dir):
        for filename in os.listdir(medical_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                models.append(f"medical/{filename[:-3]}")
    
    # Iterate through API models directory
    api_dir = os.path.join(os.path.dirname(__file__), "api_models")
    if os.path.exists(api_dir):
        for filename in os.listdir(api_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                models.append(f"api/{filename[:-3]}")
    
    return models