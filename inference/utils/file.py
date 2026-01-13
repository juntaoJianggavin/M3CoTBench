import os
import json
import yaml
import csv
from typing import Dict, List, Any, Optional, Union

def ensure_dir(dir_path: str) -> str:
    """
    Ensure the directory exists; create it if it doesn't.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file.
    
    Args:
        file_path: File path
        
    Returns:
        JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Write to a JSON file.
    
    Args:
        data: Data to write
        file_path: File path
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file.
    
    Args:
        file_path: File path
        
    Returns:
        YAML data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Write to a YAML file.
    
    Args:
        data: Data to write
        file_path: File path
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def write_csv(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None) -> None:
    """
    Write to a CSV file.
    
    Args:
        data: Data to write (list of dictionaries)
        file_path: File path
        fieldnames: List of field names
    """
    ensure_dir(os.path.dirname(file_path))
    
    # If fieldnames are not provided, use all keys from the first record
    if fieldnames is None and data:
        fieldnames = list(data[0].keys())
    
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def get_file_extension(file_path: str) -> str:
    """
    Get file extension.
    
    Args:
        file_path: File path
        
    Returns:
        File extension
    """
    return os.path.splitext(file_path)[1].lower()

def list_files(dir_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    List files in a directory.
    
    Args:
        dir_path: Directory path
        extensions: List of file extensions to filter by
        
    Returns:
        List of file paths
    """
    if not os.path.exists(dir_path):
        return []
    
    files = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath):
            if extensions is None or get_file_extension(filepath) in extensions:
                files.append(filepath)
    
    return files