from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import os

# [Critical Fix] Select the correct class based on the model
from transformers import (
    Qwen3VLForConditionalGeneration,      # Use this for 8B
    Qwen3VLMoeForConditionalGeneration,   # Use this for 30B MoE
    AutoProcessor
)
from accelerate import Accelerator

from models.base_model import BaseModel


class Qwen3VLModel(BaseModel):
    """Qwen3-VL multimodal model implementation (supports Standard and MoE versions)"""

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
        """Load Qwen3-VL model"""
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        print(f"Starting to load model: {model_path}")
        
        # Assign appropriate device map based on process
        if self.accelerator.num_processes > 1:
            device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            device_map = "auto"
        
        # [Critical Fix] Select the correct model class based on model name
        is_moe = "moe" in self.model_name.lower() or "30b" in self.model_name.lower()
        model_class = Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration
        
        print(f"Detected model type: {'MoE' if is_moe else 'Standard'}")
        print(f"Using model class: {model_class.__name__}")
        
        # Load model
        try:
            self.model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device_map,
                trust_remote_code=True,
            )
            print("✓ Model loaded successfully using flash_attention_2")
        except Exception as e:
            print(f"⚠️ Failed to load with flash_attention_2: {e}")
            print("Falling back to default attention implementation...")
            self.model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
            print("✓ Model loaded successfully using default attention")
        
        print(f"Model loading completed - Type: {type(self.model).__name__}")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Processor loaded")
        
        # Extract tokenizer
        self.tokenizer = self.processor.tokenizer
        print(f"✓ Tokenizer extracted: {type(self.tokenizer).__name__}")
        print(f"  Vocab size: {self.tokenizer.vocab_size}")
        print(f"  PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"  EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
        self.eval()

    def generate(self, image: Union[Image.Image, str, List[Union[Image.Image, str]]], 
                 prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Generate text response - Implemented exactly following official examples
        """
        # Determine if it is batch processing
        is_batch = isinstance(image, list)
        
        print("\n" + "="*50)
        print(f"Process {self.accelerator.process_index} - Starting inference")
        
        # Prepare message format (Official format)
        if is_batch:
            assert isinstance(prompt, list) and len(image) == len(prompt)
            all_messages = []
            for img, p in zip(image, prompt):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": p}
                    ]
                }]
                all_messages.append(messages)
        else:
            if isinstance(image, Image.Image):
                print(f"Input image size: {image.size}")
            print(f"Input prompt: {prompt[:80]}...")
            
            all_messages = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]]
        
        # Process each sample
        output_texts = []
        for idx, messages in enumerate(all_messages):
            if len(all_messages) > 1:
                print(f"\nProcessing sample {idx+1}/{len(all_messages)}")
            
            # [Official Example] Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # [Official Example] Move to correct device
            inputs = inputs.to(self.model.device)
            
            # [Official Example] Generate
            max_new_tokens = kwargs.pop("max_new_tokens", self._get_max_new_tokens())
            print(f"Generation params: max_new_tokens={max_new_tokens}")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
            
            print(f"Generation completed: {generated_ids.shape}")
            
            # [Official Example] Trim input section
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # [Official Example] Decode
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            output_texts.extend(output_text)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return results
        if is_batch:
            print(f"\nGenerated {len(output_texts)} results")
            if output_texts:
                print(f"Example: {output_texts[0][:150]}...")
            return output_texts
        else:
            result = output_texts[0] if output_texts else ""
            print(f"\nGenerated Result:\n{'-'*50}\n{result[:300]}\n{'-'*50}")
            return result

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()