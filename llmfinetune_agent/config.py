"""Module for managing configurations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json

@dataclass
class ModelConfig:
    """Configuration for the model."""
    base_model_name: str
    model_type: str
    max_length: int = 2048
    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    device_map: str = "auto"

class ConfigManager:
    """Manager for handling configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with path.open("r") as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to a JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w") as f:
            json.dump(config, f, indent=2)
