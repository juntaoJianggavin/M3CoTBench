# models/internvl.py

from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import math
import numpy as np
from PIL import Image
import os
import logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator

from models.base_model import BaseModel

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Configure logger
logger = logging.getLogger(__name__)

class InternVLModel(BaseModel):
    """InternVL Multimodal Model Implementation"""

    def __init__(self, model_name: str, model_config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(model_name, model_config, device)
        
        # Get model configuration parameters
        self.max_image_size = model_config.get("max_image_size", 448)
        self.image_size = model_config.get("image_size", 448)
        self.max_num_tiles = model_config.get("max_num_tiles", 12)
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set device based on accelerator
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            print(f"Multi-GPU environment: Process {self.accelerator.process_index} using GPU {self.accelerator.local_process_index}")
        else:
            self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Single-GPU environment: Using device {self._device}")
            
        # Initialize transforms
        self.transform = self._build_transform(self.image_size)
        
        # Load model
        self.load()
    
    def _build_transform(self, input_size):
        """Build image transformations"""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the aspect ratio closest to the original image"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamic image preprocessing - split image based on aspect ratio"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate possible split ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the aspect ratio closest to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == blocks
        
        # Add thumbnail (for global context)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def _process_image(self, image: Union[Image.Image, str], max_num=12):
        """Process image into model input format"""
        # If it is a path, load the image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')
        
        # Ensure it is a PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Perform dynamic preprocessing
        images = self._dynamic_preprocess(
            image, 
            image_size=self.image_size, 
            use_thumbnail=True, 
            max_num=max_num
        )
        
        # Apply transformations
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        return pixel_values
    
    def _process_video(self, video_path: str, bound=None, num_segments=8, max_num=1):
        """Process video as model input"""
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("decord library is required to process video: pip install decord")
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        # Calculate frame indices
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        
        start_idx = max(0, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        
        # Process each frame
        pixel_values_list = []
        num_patches_list = []
        
        for frame_index in frame_indices:
            # Get frame and convert to PIL image
            frame = vr[frame_index].asnumpy()
            img = Image.fromarray(frame).convert('RGB')
            
            # Dynamic frame preprocessing
            img_patches = self._dynamic_preprocess(
                img, 
                image_size=self.image_size, 
                use_thumbnail=True, 
                max_num=max_num
            )
            pixel_values = [self.transform(tile) for tile in img_patches]
            pixel_values = torch.stack(pixel_values)
            
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        # Concatenate all frames
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    def _split_model(self, model_name):
        """Split model across multiple GPUs"""
        device_map = {}
        world_size = torch.cuda.device_count()
        
        if world_size <= 1:
            return "auto"
        
        # Get number of layers based on model name
        model_size = model_name.split('-')[-1] if '-' in model_name else "8B"
        # Remove possible 'B' suffix
        model_size = model_size.replace('B', '')
        
        num_layers = {
            '1': 24, '2': 24, '4': 36, '8': 32,
            '26': 48, '38': 64, '78': 80
        }.get(model_size, 32)
        
        print(f"Model {model_name} has {num_layers} layers")
        
        # The first GPU is used for ViT, so treat it as half a GPU
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(min(num_layer, num_layers - layer_cnt)):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
                if layer_cnt >= num_layers:
                    break
        
        # Assign other components
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        
        return device_map

    def load(self) -> None:
        """Load InternVL model"""
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        print(f"Starting to load model: {model_path}")
        
        load_in_8bit = self.model_config.get("load_in_8bit", False)
        
        # ✅ Fix: Use simple device mapping strategy
        if self.accelerator.num_processes > 1:
            # Multi-process DDP mode: Each process uses its own GPU
            device_map = f"cuda:{self.accelerator.local_process_index}"
            print(f"Multi-GPU DDP mode: Process {self.accelerator.process_index} -> {device_map}")
        else:
            # Single-process mode: Use 'auto' for automatic allocation (supports model parallelism)
            device_map = "auto"
            print(f"Single-GPU mode, using automatic device mapping")
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=True,
            use_flash_attn=self.model_config.get("use_flash_attn", True),
            trust_remote_code=True,
            device_map=device_map  # ✅ Use corrected device_map
        ).eval()
        
        print("Model loading completed")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        print("Tokenizer loading completed")
        print(f"Current process device: {self._device}")
        print("All loaded successfully.")

    def generate(self, image: Union[Image.Image, str, List[Union[Image.Image, str]]], 
                 prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Generate text response
        
        Args:
            image: Single image, image path, or list of images
            prompt: Single prompt or list of prompts
            **kwargs: Other generation parameters
                - is_video: Whether it is video input
                - video_path: Video path (when is_video=True)
                - num_segments: Number of video frames (default 8)
                - bound: Video time bound (start, end)
        
        Returns:
            Generated text or list of texts
        """
        # Determine if batch processing
        is_batch = isinstance(image, list) and isinstance(prompt, list)
        is_video = kwargs.pop("is_video", False)
        
        print("\n" + "="*50)
        print(f"Process {self.accelerator.process_index} - Starting inference")
        
        # Generation config
        generation_config = {
            "max_new_tokens": kwargs.pop("max_new_tokens", self._get_max_new_tokens()),
            "do_sample": kwargs.pop("do_sample", True),
        }
        
        # Add temperature parameter only when do_sample=True
        if generation_config["do_sample"]:
            generation_config["temperature"] = kwargs.pop("temperature", self._get_temperature())
            generation_config["top_p"] = kwargs.pop("top_p", 0.9)
        
        generation_config.update(kwargs)
        
        try:
            if is_video:
                # Video processing
                video_path = kwargs.pop("video_path", image if isinstance(image, str) else None)
                if not video_path:
                    raise ValueError("Video mode requires 'video_path' parameter")
                
                pixel_values, num_patches_list = self._process_video(
                    video_path,
                    bound=kwargs.pop("bound", None),
                    num_segments=kwargs.pop("num_segments", 8),
                    max_num=kwargs.pop("max_num", 1)
                )
                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)
                
                # Generate video frame prefix
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                question = video_prefix + prompt
                
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
                
                print("\nGenerated Results:")
                print("-"*50)
                print(response)
                print("-"*50)
                
                return response
                
            elif is_batch:
                # Batch image processing
                if len(image) != len(prompt):
                    raise ValueError(f"Number of images ({len(image)}) does not match number of prompts ({len(prompt)})")
                
                print(f"Batch processing: {len(image)} samples")
                
                # Process all images
                all_pixel_values = []
                all_num_patches = []
                
                for img in image:
                    pixel_values = self._process_image(img, max_num=self.max_num_tiles)
                    all_pixel_values.append(pixel_values)
                    all_num_patches.append(pixel_values.shape[0])
                
                # Concatenate
                pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16).to(self._device)
                
                # Add <image> token to each prompt
                questions = [
                    f'<image>\n{p}' if not p.startswith('<image>') else p
                    for p in prompt
                ]
                
                # Batch generation
                responses = self.model.batch_chat(
                    self.tokenizer,
                    pixel_values,
                    num_patches_list=all_num_patches,
                    questions=questions,
                    generation_config=generation_config
                )
                
                print(f"\nBatch generation completed, total {len(responses)} results")
                print("First result example:")
                print("-"*50)
                print(responses[0] if responses else "No result")
                print("-"*50)
                
                return responses
                
            else:
                # Single image processing
                if isinstance(image, Image.Image):
                    print(f"Input image size: {image.size}")
                elif isinstance(image, str):
                    print(f"Input image path: {image}")
                
                pixel_values = self._process_image(image, max_num=self.max_num_tiles)
                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)
                
                # Add <image> token
                question = f'<image>\n{prompt}' if not prompt.startswith('<image>') else prompt
                
                response, _ = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True
                )
                
                print("\nGenerated Results:")
                print("-"*50)
                print(response)
                print("-"*50)
                
                return response
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            
            if is_batch:
                return [f"Generation error: {str(e)}"] * len(prompt)
            else:
                return f"Generation error: {str(e)}"
        finally:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()