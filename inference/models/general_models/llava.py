from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import os
import copy
from accelerate import Accelerator

from models.base_model import BaseModel

# Use official LLaVA library
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

class LlavaOnevisionModel(BaseModel):
    """LLaVA OneVision multimodal model implementation (using official inference pipeline)"""

    def __init__(self, model_name: str, model_config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(model_name, model_config, device)
        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            print(f"Multi-GPU environment: Process {self.accelerator.process_index} using GPU {self.accelerator.local_process_index}")
        else:
            self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Single-GPU environment: Using device {self._device}")
        self.load()

    def load(self) -> None:
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        print(f"Starting to load model: {model_path}")
        
        # Monkey patch to disable Flash Attention
        import transformers.models.qwen2.modeling_qwen2 as qwen2_module
        
        # Save original forward method
        if hasattr(qwen2_module, 'Qwen2FlashAttention2'):
            original_forward = qwen2_module.Qwen2FlashAttention2.forward
            
            # Replace with standard attention forward
            if hasattr(qwen2_module, 'Qwen2Attention'):
                qwen2_module.Qwen2FlashAttention2.forward = qwen2_module.Qwen2Attention.forward
        
        model_name = "llava_qwen"
        
        if self.accelerator.num_processes > 1:
            device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            device_map = "auto"
        
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path, 
            None, 
            model_name, 
            device_map=device_map,
            torch_dtype=torch.bfloat16
        )
        
        def convert_to_bf16(module):
            for param in module.parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.bfloat16)
            for buffer_name, buffer in module.named_buffers():
                if buffer.dtype == torch.float32:
                    module._buffers[buffer_name] = buffer.to(torch.bfloat16)
        
        convert_to_bf16(self.model)
        self.model.eval()
        
        self.conv_template = "qwen_1_5"
        
        print("Model loading completed")
        print(f"Current process device: {self._device}")
        print(f"Model data type: {next(self.model.parameters()).dtype}")
        print("All loaded successfully.")

    def generate(
        self, 
        image: Union[Image.Image, str, List[Union[Image.Image, str]]], 
        prompt: Union[str, List[str]], 
        **kwargs
    ) -> Union[str, List[str]]:
        is_batch = isinstance(image, list)
        
        if is_batch:
            assert isinstance(prompt, list) and len(image) == len(prompt), "Number of images and prompts must match in batch mode"
            pil_images = []
            for img in image:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_images.append(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            prompts = prompt
        else:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            pil_images = [image]
            prompts = [prompt]
        
        image_tensors = process_images(pil_images, self.image_processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.bfloat16, device=self._device) for _image in image_tensors]
        
        all_input_ids = []
        all_image_sizes = []
        
        for img, text in zip(pil_images, prompts):
            conv = copy.deepcopy(conv_templates[self.conv_template])
            question = DEFAULT_IMAGE_TOKEN + "\n" + text
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt_question, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            ).unsqueeze(0).to(self._device)
            
            all_input_ids.append(input_ids)
            all_image_sizes.append(img.size)
        
        if len(all_input_ids) == 1:
            input_ids = all_input_ids[0]
            image_sizes = all_image_sizes
        else:
            max_len = max(ids.shape[1] for ids in all_input_ids)
            padded_input_ids = []
            for ids in all_input_ids:
                padding_length = max_len - ids.shape[1]
                if padding_length > 0:
                    padded = torch.cat([
                        ids,
                        torch.full((1, padding_length), self.tokenizer.pad_token_id, 
                                   dtype=ids.dtype, device=ids.device)
                    ], dim=1)
                else:
                    padded = ids
                padded_input_ids.append(padded)
            input_ids = torch.cat(padded_input_ids, dim=0)
            image_sizes = all_image_sizes
        
        gen_kwargs = {
            "max_new_tokens": kwargs.pop("max_new_tokens", self._get_max_new_tokens()),
            "temperature": kwargs.pop("temperature", self._get_temperature()),
            "do_sample": kwargs.pop("do_sample", None),
            **kwargs
        }
        
        if gen_kwargs["temperature"] > 0 and gen_kwargs["do_sample"] is None:
            gen_kwargs["do_sample"] = True
        elif gen_kwargs["temperature"] == 0:
            gen_kwargs["do_sample"] = False
        
        if gen_kwargs["do_sample"] is None:
            del gen_kwargs["do_sample"]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    **gen_kwargs
                )
        
        text_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        results = []
        for i, output in enumerate(text_outputs):
            if "ASSISTANT:" in output:
                answer = output.split("ASSISTANT:")[-1].strip()
            elif "assistant" in output.lower():
                answer = output.split("assistant")[-1].strip()
            else:
                answer = output.strip()
            results.append(answer)
        
        if not is_batch:
            results = results[0]
            print("\nGenerated Results:\n" + "-"*50)
            print(results)
            print("-"*50)
        else:
            print("\nGenerated Result Example:")
            print("-"*50)
            print(results[0] if results else "No result")
            print("-"*50)
            print(f"(Generated {len(results)} results in total)")
        
        print(f"Process {self.accelerator.process_index} - Inference finished - Generated {len(results) if is_batch else 1} results")
        return results