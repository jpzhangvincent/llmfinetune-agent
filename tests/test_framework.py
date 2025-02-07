"""Test script for the LLM fine-tuning framework."""
import json
import logging
import os
from pathlib import Path

import pytest
from datasets import Dataset

from llmfinetune_agent.agents.utils import load_config
from llmfinetune_agent.agents.orchestrator import OrchestratorAgent
from llmfinetune_agent.agents.data_generator import DataGeneratorAgent
from llmfinetune_agent.agents.trainer_agent import TrainerAgent
from llmfinetune_agent.agents.evaluator import EvaluatorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    """Load test configuration."""
    config = load_config("configs/default_config.json")
    
    # Update config for testing
    config.update({
        "model": {
            "base_model_name": "meta-llama/Llama-2-7b-hf",
            "model_type": "causal_lm",
            "max_length": 512,
            "torch_dtype": "float16",
            "load_in_4bit": True
        },
        "training": {
            "output_dir": "test_outputs",
            "num_train_epochs": 1,  # Reduced for testing
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
            "max_grad_norm": 0.3,
            "warmup_ratio": 0.03
        },
        "data": {
            "train_file": "data/train.json",
            "validation_file": "data/validation.json",
            "text_column": "text"
        },
        "evaluation": {
            "metrics": ["rouge", "bleu"],
            "thresholds": {
                "rouge1": 0.4,
                "bleu": 0.3
            }
        }
    })
    return config

def load_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return Dataset.from_list(data)

def test_data_generator(config):
    """Test the data generator agent."""
    agent = DataGeneratorAgent(config)
    
    # Load sample data
    train_data = load_dataset(config["data"]["train_file"])
    val_data = load_dataset(config["data"]["validation_file"])
    
    # Update agent state with data
    agent.state.config.update({
        "train_dataset": train_data,
        "eval_dataset": val_data
    })
    
    # Run data generation
    state = agent.run()
    
    assert state.completed
    assert state.error is None
    assert state.dataset is not None
    assert "train" in state.dataset
    if config["data"].get("validation_file"):
        assert "validation" in state.dataset

def test_training_workflow(config):
    """Test the complete training workflow."""
    # Initialize agents
    data_agent = DataGeneratorAgent(config)
    trainer_agent = TrainerAgent(config)
    evaluator_agent = EvaluatorAgent(config)
    
    # Prepare data
    train_data = load_dataset(config["data"]["train_file"])
    val_data = load_dataset(config["data"]["validation_file"])
    
    data_agent.state.config.update({
        "train_dataset": train_data,
        "eval_dataset": val_data
    })
    
    # Run data preparation
    data_state = data_agent.run()
    assert data_state.completed
    
    # Update trainer state with prepared data
    trainer_agent.state.config.update({
        "train_dataset": data_state.dataset["train"],
        "eval_dataset": data_state.dataset["validation"]
    })
    
    # Run training
    trainer_state = trainer_agent.run()
    assert trainer_state.completed
    assert trainer_state.model_path is not None
    
    # Update evaluator state
    evaluator_agent.state.model_path = trainer_state.model_path
    evaluator_agent.state.config.update({
        "eval_dataset": data_state.dataset["validation"]
    })
    
    # Run evaluation
    eval_state = evaluator_agent.run()
    assert eval_state.completed
    
    # Check metrics
    metrics = eval_state.metrics_results
    assert metrics is not None
    assert "rouge" in metrics
    if "bleu" in config["evaluation"]["metrics"]:
        assert "bleu" in metrics
        
def test_orchestrator(config):
    """Test the orchestrator agent."""
    orchestrator = OrchestratorAgent(config)
    
    # Prepare initial state
    train_data = load_dataset(config["data"]["train_file"])
    val_data = load_dataset(config["data"]["validation_file"])
    
    orchestrator.state.config.update({
        "train_dataset": train_data,
        "eval_dataset": val_data
    })
    
    # Run orchestrator
    state = orchestrator.run()
    
    assert state.completed
    assert state.error is None
    assert state.current_stage.value == "completed"

def test_rlhf_workflow(config):
    """Test RLHF training workflow."""
    # Update config for RLHF
    config["training"]["use_rlhf"] = True
    
    # Initialize agents
    data_agent = DataGeneratorAgent(config)
    trainer_agent = TrainerAgent(config)
    
    # Prepare data
    train_data = load_dataset(config["data"]["train_file"])
    
    data_agent.state.config.update({
        "train_dataset": train_data
    })
    
    # Run data preparation
    data_state = data_agent.run()
    assert data_state.completed
    
    # Update trainer state
    trainer_agent.state.config.update({
        "train_dataset": data_state.dataset["train"],
        "use_rlhf": True
    })
    
    # Run RLHF training
    trainer_state = trainer_agent.run()
    assert trainer_state.completed
    assert trainer_state.model_path is not None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
