from typing import Dict, Any, Optional, List, Union
import torch
from PIL import Image
import os
import re

from transformers import MllamaForConditionalGeneration, AutoProcessor
from accelerate import Accelerator

from models.base_model import BaseModel

class LlamaVisionModel(BaseModel):
    """Llama-3-Vision-Instruct multimodal model, Qwen interface style"""

    def __init__(self, model_name: str, model_config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(model_name, model_config, device)
        self.accelerator = Accelerator()

        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
        else:
            self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load()

    def load(self) -> None:
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        if self.accelerator.num_processes > 1:
            device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            device_map = "auto"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.eval()

    @staticmethod
    def extract_assistant_answer(text: str) -> str:
        """
        Extracts the assistant response, removing conversation templates/headers and other redundancy, 
        returning only the main body of the model's answer.
        """
        match = re.search(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n*(.*?)(?=<\|eot_id\|>|$)",
            text,
            re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return text.strip()

    def generate(self, image: Union[Image.Image, str, List[Union[Image.Image, str]]],
                 prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        image: Single image object or path, or a list of them.
        prompt: string or list of strings.
        """
        is_batch = isinstance(image, list)
        if is_batch:
            batch_size = len(image)
            assert isinstance(prompt, list) and len(prompt) == batch_size, "Image and text lengths must match"
            messages = [
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": p}
                ]}] for p in prompt
            ]
            input_texts = [self.processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
            images = [img if isinstance(img, Image.Image) else Image.open(img) for img in image]
            inputs = self.processor(
                images,
                input_texts,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self._device)
        else:
            img = image if isinstance(image, Image.Image) else Image.open(image)
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                img,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self._device)

        gen_kwargs = {
            "max_new_tokens": kwargs.pop("max_new_tokens", self._get_max_new_tokens()), 
            "temperature": kwargs.pop("temperature", self._get_temperature()),
            # ...
            **kwargs
        }

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        if is_batch:
            # output: (batch, seq)
            results = [self.extract_assistant_answer(self.processor.decode(out)) for out in output]
            return results
        else:
            return self.extract_assistant_answer(self.processor.decode(output[0]))