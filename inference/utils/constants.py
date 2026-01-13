"""Constant definitions"""

# Image related
DEFAULT_IMAGE_SIZE = 336

# Model related
MODEL_TYPES = [
    "general",  # General Vision-Language Model
    "medical",  # Medical-specific Vision-Language Model
    "api"       # API-based model
]

# Evaluation modes
EVAL_MODES = [
    "cot",      # Chain-of-Thought reasoning
    "direct"    # Direct reasoning
]

# Output formats
OUTPUT_FORMATS = [
    "json",     # JSON format
    "csv",      # CSV format
    "txt"       # Plain text format
]

# Default configuration file path
DEFAULT_CONFIG_PATH = "config/models.yaml"

# Inference engine settings
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 2