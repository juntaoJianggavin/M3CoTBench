# medical_models/medgemma.py

from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import os

from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator

from models.base_model import BaseModel


class MedGemmaModel(BaseModel):
    """MedGemma multimodal medical model implementation."""

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
        """Load MedGemma model."""
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        print(f"Starting to load model: {model_path}")
        
        # Assign appropriate device map based on process
        if self.accelerator.num_processes > 1:
            # In multi-card mode, each process uses the specified GPU
            device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            # Keep as is for single-card mode
            device_map = "auto"
            
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device_map,
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
        Generate text response.
        Each process handles only its own batch; does not attempt cross-process batching.
        """
        # Determine if it is batch processing
        is_batch = isinstance(image, list)
        
        # Ensure handling single sample per process in multi-GPU setup if needed
        if self.accelerator.num_processes > 1 and is_batch and len(image) > 1:
            print(f"Warning: Processing single sample per iteration is recommended in multi-GPU environment")
            
        print("\n" + "="*50)
        print(f"Process {self.accelerator.process_index} - Starting inference - {'Batch processing' if is_batch else 'Single sample'}")
        print(f"Processing input - Text count: {len(prompt) if is_batch else 1}, Image count: {len(image) if is_batch else 1}")
        
        # Get system prompt (MedGemma feature: Expert Radiologist)
        system_prompt = self._get_system_prompt()
        
        # Prepare message format
        if is_batch:
            # Ensure consistent batch size
            batch_size = len(image)
            assert isinstance(prompt, list) and len(image) == len(prompt), "Batch processing requires same length of images and prompts"
            
            # Create message list for each sample
            messages_list = []
            for img, p in zip(image, prompt):
                # Load image (if path)
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": p},
                            {"type": "image", "image": img}
                        ]
                    }
                ]
                messages_list.append(messages)
        else:
            # Single sample processing
            if isinstance(image, str):
                print(f"Input image path: {image}")
                image = Image.open(image).convert("RGB")
            
            if isinstance(image, Image.Image):
                print(f"Input image size: {image.size}")
                
            print(f"Input prompt: {prompt[:100]}..." if len(prompt) > 100 else prompt)
            
            # Create MedGemma format messages
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            messages_list = [messages]
        
        print(f"Starting generation - Max new tokens: {kwargs.get('max_new_tokens', self._get_max_new_tokens())}")
        
        # Process all messages in batch
        all_outputs = []
        for messages in messages_list:
            # Apply chat template and prepare inputs
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": kwargs.pop("max_new_tokens", self._get_max_new_tokens()),
                "do_sample": kwargs.pop("do_sample", False),  # MedGemma defaults to greedy decoding
                **kwargs
            }

            # Generate
            with torch.inference_mode():
                generation = self.model.generate(**inputs, **gen_kwargs)
                # Keep only newly generated tokens
                generation = generation[0][input_len:]
            
            # Decode
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            all_outputs.append(decoded)
        
        # Clear cache to prevent memory leaks
        torch.cuda.empty_cache()
        
        # Return results
        if is_batch:
            result = all_outputs
            
            # Print the first result as an example
            print("\nGenerated Result Example:")
            print("-"*50)
            print(result[0] if result else "No result")
            print("-"*50)
            print(f"(Generated {len(result)} results in total)")
        else:
            # Single result
            result = all_outputs[0] if all_outputs else ""
            
            print("\nGenerated Result:")
            print("-"*50)
            print(result)
            print("-"*50)
            
        print(f"Process {self.accelerator.process_index} - Inference finished - Generated {len(all_outputs)} results")
            
        return result

    def _get_system_prompt(self) -> str:
        """
        Get system prompt.
        MedGemma defaults to using an expert radiologist persona.
        """
        # Can be read from config, otherwise use default
        return self.model_config.get('system_prompt', "You are an expert radiologist.")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()