"""Shared utility functions for agents."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AgentState:
    """Base state class for all agents."""
    config: Dict[str, Any]
    messages: list[str] = None
    error: Optional[str] = None
    completed: bool = False
    current_stage: Optional[str] = None
    artifacts: Dict[str, Any] = None
    metrics: Dict[str, float] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.messages is None:
            self.messages = []
        if self.artifacts is None:
            self.artifacts = {}
        if self.metrics is None:
            self.metrics = {}

def load_config(config_path: str = "configs/default_config.json") -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    return config

def format_agent_message(agent_name: str, message: str) -> str:
    """Format a message from an agent."""
    return f"[{agent_name.upper()}] {message}"

class AgentException(Exception):
    """Base exception class for agent errors."""
    pass

def validate_model_name(model_name: str) -> bool:
    """Validate that a model name is properly formatted."""
    parts = model_name.split("/")
    return len(parts) == 2 and all(parts)

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save agent state checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)

def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load agent state checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    with open(path) as f:
        state = json.load(f)
    return state
