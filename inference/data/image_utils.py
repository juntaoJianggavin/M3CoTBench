import os
from typing import Optional, Callable, Union, Tuple
import torch
from PIL import Image
import numpy as np

def load_and_process_image(
    image_path: str,
    transform: Optional[Callable] = None,
    max_image_size: int = 512,  # New max edge length parameter, default 512
    logger=None                 # Optional logger object
) -> Union[Image.Image, torch.Tensor]:
    """
    Load and process an image, and automatically limit the maximum size.
    
    Args:
        image_path: Path to the image.
        transform: Image transformation function.
        max_image_size: Maximum edge length of the image.
        logger: Optional logger object.
        
    Returns:
        Processed image.
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Attempt to load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    # Limit maximum size
    width, height = image.size
    if max(width, height) > max_image_size:
        scaling_factor = max_image_size / float(max(width, height))
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        if logger is not None:
            logger.info(f"Image {image_path} is resized to {new_width}x{new_height}")
        else:
            print(f"Image {image_path} is resized to {new_width}x{new_height}")
    
    # Apply transformation (if any)
    if transform is not None:
        image = transform(image)
        
    return image

def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (336, 336),
    keep_aspect_ratio: bool = True
) -> Image.Image:
    """
    Resize image.
    
    Args:
        image: Input image.
        target_size: Target size (width, height).
        keep_aspect_ratio: Whether to keep aspect ratio.
        
    Returns:
        Resized image.
    """
    if keep_aspect_ratio:
        # Resize while keeping aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        target_width, target_height = target_size
        
        if width > height:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
            
        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size (black background)
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        
        # Calculate paste position (centered)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste the resized image
        new_image.paste(image, (paste_x, paste_y))
        return new_image
    else:
        # Resize directly to target size
        return image.resize(target_size, Image.Resampling.LANCZOS)

def normalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
) -> torch.Tensor:
    """
    Normalize image tensor.
    
    Args:
        image: Image tensor [C, H, W].
        mean: Mean.
        std: Standard deviation.
        
    Returns:
        Normalized image tensor.
    """
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image - mean) - std