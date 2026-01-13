import os
import json
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import string

from data.image_utils import load_and_process_image
from data.prompt_templates import get_cot_prompt, get_direct_prompt, extract_options_from_doc

class MedicalVQADataset(Dataset):
    """Medical Visual Question Answering Dataset"""

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        mode: str = 'cot',
        transform=None,
        pre_prompt: str = "",
        post_prompt: str = "",
        max_image_size: int = 512
    ):
        self.image_dir = image_dir
        self.mode = mode.lower()
        self.transform = transform
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.max_image_size = max_image_size

        if self.mode not in ['cot', 'direct']:
            raise ValueError(f"Mode {mode} not supported. Use 'cot' or 'direct'")

        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                # Convert to list format
                if isinstance(raw_data, dict):
                    if "data" in raw_data:
                        raw_data = raw_data["data"]
                    else:
                        raw_data = list(raw_data.values())
            self.data = [
                {'index': str(item['index']), 'question': str(item['question'])}
                for item in raw_data
                if 'index' in item and 'question' in item
            ]
        elif file_ext == '.csv':
            df = pd.read_csv(data_path)
            if not all(col in df.columns for col in ['index', 'question']):
                raise ValueError("CSV must contain 'index' and 'question' columns")
            self.data = df[['index', 'question']].astype(str).to_dict(orient='records')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
            if not all(col in df.columns for col in ['index', 'question']):
                raise ValueError("Excel must contain 'index' and 'question' columns")
            self.data = df[['index', 'question']].astype(str).to_dict(orient='records')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .json, .csv, or .xlsx")
   
    def _load_csv_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Load data in CSV format.
        
        Args:
            csv_path: Path to the CSV file.
            
        Returns:
            List of data dictionaries.
        """
        # Read CSV using pandas
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ['index', 'question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        
        # Convert to list of dictionaries
        data_list = []
        
        for _, row in df.iterrows():
            item = row.to_dict()
            
            # Handle 'key_annotation_steps' column, which might contain a JSON string
            if 'key_annotation_steps' in item and isinstance(item['key_annotation_steps'], str):
                try:
                    item['key_annotation_steps'] = json.loads(item['key_annotation_steps'])
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, keep the original string
                    pass
            
            data_list.append(item)
        
        return data_list
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        index = item['index']

        # Find image path
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.webp']:
            potential_path = os.path.join(self.image_dir, f"{index}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None or not os.path.exists(image_path):
            print(f"Warning: Image not found for index {index}")
            placeholder_image = Image.new('RGB', (336, 336), color='black')
            image = self.transform(placeholder_image) if self.transform else placeholder_image
        else:
            image = load_and_process_image(image_path, self.transform, max_image_size=self.max_image_size)

        # Extract question text and options
        question = item.get('question', '')
        if self.pre_prompt:
            question = f"{self.pre_prompt}{question}"
        if self.post_prompt:
            question = f"{question}{self.post_prompt}"

        options = {}
        question_lines = question.split('\n')
        for line in question_lines[1:]:
            for letter in string.ascii_uppercase:
                if line.strip().startswith(f"{letter}.") or line.strip().startswith(f"{letter})"):
                    option_text = line.strip()[3:].strip()
                    options[letter] = option_text

        if not options:
            options = extract_options_from_doc(item)

        prompt = (
            get_cot_prompt(question, options) if self.mode == 'cot'
            else get_direct_prompt(question, options)
        )

        return {
            'id': index,
            'image': image,
            'question': question,
            'prompt': prompt,
            'answer': '',  # No longer reading 'answer' and other fields
            'options': options,
            'cot_reference': None,
            'category': '',
            'question_type': '',
            'examination_type': '',
            'metadata': {}
        }

def create_dataloader(
    dataset: MedicalVQADataset,
    batch_size: int = 1,
    num_workers: int = 2,
    shuffle: bool = False,
    drop_last=False
) -> DataLoader:
    """
    Create data loader.
    
    Args:
        dataset: Medical VQA dataset.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle data.
        
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch: Batch data.
        
    Returns:
        Batched data.
    """
    # Separate fields
    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    answers = [item['answer'] for item in batch]
    options = [item['options'] for item in batch]
    categories = [item['category'] for item in batch]
    question_types = [item['question_type'] for item in batch]
    examination_types = [item['examination_type'] for item in batch]
    cot_references = [item['cot_reference'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    # Return batch in dictionary format
    return {
        'id': ids,
        'image': images,
        'question': questions,
        'prompt': prompts,
        'answer': answers,
        'options': options,
        'category': categories,
        'question_type': question_types,
        'examination_type': examination_types,
        'cot_reference': cot_references,
        'metadata': metadata
    }