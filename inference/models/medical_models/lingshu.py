# medical_models/lingshu.py

from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

from models.base_model import BaseModel


class LingshuModel(BaseModel):
    """Lingshu Medical Multimodal Model Implementation (based on Qwen2.5-VL architecture)"""

    def __init__(self, model_name: str, model_config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(model_name, model_config, device)
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Set device based on accelerator
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            print(f"Multi-GPU environment: Process {self.accelerator.process_index} using GPU {self.accelerator.local_process_index}")
        else:
            self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Single-GPU environment: Using device {self._device}")
            
        self.load()

    def load(self) -> None:
        """Load Lingshu model"""
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        print(f"Starting to load model: {model_path}")
        
        # Assign appropriate device map based on process
        if self.accelerator.num_processes > 1:
            # In multi-GPU mode, each process uses the specified GPU
            device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            # Keep as is for single-GPU mode
            device_map = "auto"
        
        # Check if flash_attention_2 is supported
        use_flash_attn = self.model_config.get("use_flash_attention", True)
        attn_implementation = "flash_attention_2" if use_flash_attn else None
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map=device_map,
                ignore_mismatched_sizes=True,
            )
            if use_flash_attn:
                print("Flash Attention 2 enabled for performance improvement")
        except Exception as e:
            print(f"Failed to load Flash Attention 2, falling back to default attention mechanism: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                ignore_mismatched_sizes=True,
            )
        
        print("Model loading completed")
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Processor loading completed")
        
        print(f"Current process device: {self._device}")
        print("All loaded successfully.")
        self.eval()

    def generate(self, image: Union[Image.Image, str, List[Union[Image.Image, str]]], 
                prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Generate text response
        Each process handles only its own batch; does not attempt cross-process batching
        """
        # Determine if it is batch processing
        is_batch = isinstance(image, list)
        
        # Ensure processing single sample
        if self.accelerator.num_processes > 1 and is_batch and len(image) > 1:
            print(f"Warning: Processing single sample per iteration is recommended in multi-GPU environment")
            
        print("\n" + "="*50)
        print(f"Process {self.accelerator.process_index} - Starting inference - {'Batch processing' if is_batch else 'Single sample'}")
        print(f"Processing input - Text count: {len(prompt) if is_batch else 1}, Image count: {len(image) if is_batch else 1}")
        
        # Prepare message format
        if is_batch:
            # Ensure batch size consistency
            batch_size = len(image)
            assert isinstance(prompt, list) and len(image) == len(prompt), "Batch processing requires same length of images and prompts"
            
            # Create a message list for each sample
            messages = []
            for i, (img, p) in enumerate(zip(image, prompt)):
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": p}
                    ]
                }]
                messages.append(msg)
            
            # Batch process chat templates
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
        else:
            # Single sample processing
            if isinstance(image, Image.Image):
                print(f"Input image size: {image.size}")
            elif isinstance(image, str) and os.path.exists(image):
                print(f"Input image path: {image}")
                
            print(f"Input prompt: {prompt[:100]}..." if len(prompt) > 100 else prompt)
            
            # Create message in Lingshu format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Apply chat template
            texts = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info - messages must be wrapped in a list
            image_inputs, video_inputs = process_vision_info([messages])
        
        print(f"Starting generation - Max new tokens: {kwargs.get('max_new_tokens', self._get_max_new_tokens())}, Temperature: {kwargs.get('temperature', self._get_temperature())}")
        
        # Process input
        if is_batch:
            text_input = texts  # Already a list of strings
        else:
            text_input = [texts]  # Convert single string to list
            
        # Create model inputs using processor
        inputs = self.processor(
            text=text_input,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
                
        # Move all inputs to model device
        inputs = inputs.to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": kwargs.pop("max_new_tokens", self._get_max_new_tokens()),
            "temperature": kwargs.pop("temperature", self._get_temperature()),
            "top_p": kwargs.pop("top_p", 0.9),
            "repetition_penalty": kwargs.pop("repetition_penalty", 1.0),
            "do_sample": kwargs.pop("do_sample", None),
            **kwargs
        }
        
        # Automatically set do_sample based on temperature
        if gen_kwargs["temperature"] > 0 and gen_kwargs["do_sample"] is None:
            gen_kwargs["do_sample"] = True
        if gen_kwargs["do_sample"] is None:
            del gen_kwargs["do_sample"]

        print("Generating...")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        print("Generation completed, starting to process results")

        # Trim generated IDs (remove input part)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Clear cache to prevent memory leaks
        torch.cuda.empty_cache()
        
        # Return results
        if is_batch:
            result = output_texts
            
            # Print the first result as an example
            print("\nGenerated Result Example:")
            print("-"*50)
            print(result[0] if result else "No result")
            print("-"*50)
            print(f"(Generated {len(result)} results in total)")
        else:
            # Single result
            result = output_texts[0] if output_texts else ""
            
            print("\nGenerated Result:")
            print("-"*50)
            print(result)
            print("-"*50)
            
        print(f"Process {self.accelerator.process_index} - Inference finished - Generated {len(output_texts)} results")
            
        return result

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()