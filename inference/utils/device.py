from typing import Dict, Any
import torch
import platform
import os
import subprocess
import psutil

def get_device_info() -> Dict[str, Any]:
    """
    Get device information.
    
    Returns:
        Device information dictionary.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu": {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "name": _get_cpu_name()
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)
        }
    }
    
    # GPU information
    if torch.cuda.is_available():
        info["cuda"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2)
            }
            info["cuda"]["devices"].append(device_info)
    else:
        info["cuda"] = {"available": False}
    
    # MPS information (Apple Metal)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps"] = {"available": True}
    else:
        info["mps"] = {"available": False}
    
    return info

def _get_cpu_name() -> str:
    """
    Get CPU name.
    
    Returns:
        CPU name string.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        try:
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command.split()).strip().decode()
        except:
            return "Unknown"
    elif platform.system() == "Linux":
        try:
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command.split()).strip().decode()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return line.split(":")[1].strip()
            return "Unknown"
        except:
            return "Unknown"
    else:
        return "Unknown"

def get_optimal_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def setup_device(gpu_ids=None):
    """
    Set up the device, supporting manual GPU specification.
    
    Args:
        gpu_ids: List of GPU IDs or a comma-separated string.
        
    Returns:
        torch.device object.
    """
    # Handle GPU IDs
    if gpu_ids is not None:
        if isinstance(gpu_ids, str):
            gpu_ids = [int(id.strip()) for id in gpu_ids.split(',') if id.strip()]
        
        # Set visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(id) for id in gpu_ids)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            return torch.device(f"cuda:{0}")  # Use the first visible GPU
    
    # If not specified or CUDA is unavailable, fallback to default device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")