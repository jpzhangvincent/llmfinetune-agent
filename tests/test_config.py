"""Tests for configuration management."""


import pytest
from llmfinetune.config import ConfigManager, ModelConfig


def test_model_config():
    """Test ModelConfig initialization."""
    config = ModelConfig(
        base_model_name="mistralai/Mistral-7B-v0.1",
        model_type="causal_lm"
    )
    assert config.base_model_name == "mistralai/Mistral-7B-v0.1"
    assert config.model_type == "causal_lm"
    assert config.max_length == 2048
    assert config.load_in_4bit is True

def test_config_manager_load_save(tmp_path):
    """Test ConfigManager load and save functionality."""
    config_path = tmp_path / "test_config.json"
    test_config = {
        "model": {
            "base_model_name": "mistralai/Mistral-7B-v0.1",
            "model_type": "causal_lm"
        },
        "training": {
            "learning_rate": 2e-5,
            "num_epochs": 3
        }
    }
    
    # Test save
    ConfigManager.save_config(test_config, str(config_path))
    assert config_path.exists()
    
    # Test load
    loaded_config = ConfigManager.load_config(str(config_path))
    assert loaded_config == test_config

def test_config_manager_file_not_found():
    """Test ConfigManager handles missing files correctly."""
    with pytest.raises(FileNotFoundError):
        ConfigManager.load_config("nonexistent_config.json")
