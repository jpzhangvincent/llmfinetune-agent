"""Data generator agent for creating and preparing training data."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from .utils import AgentState, AgentException, format_agent_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataGeneratorState(AgentState):
    """State for the data generator agent."""
    input_data: Optional[Dict[str, Any]] = None
    generated_data: Optional[Dict[str, Any]] = None
    dataset: Optional[DatasetDict] = None
    data_stats: Dict[str, Any] = field(default_factory=dict)

class DataGeneratorAgent:
    """Agent responsible for generating and preparing training data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = DataGeneratorState(config=config)
        self.data_config = config.get("data", {})
        
    def run(self, state: Optional[DataGeneratorState] = None) -> DataGeneratorState:
        """Run the data generation process."""
        if state:
            self.state = state
            
        try:
            # Load and process input data
            self._load_data()
            
            # Generate synthetic data if needed
            if self.config.get("data_generation", {}).get("enabled", False):
                self._generate_synthetic_data()
            
            # Prepare final dataset
            self._prepare_dataset()
            
            # Generate data statistics
            self._generate_statistics()
            
            self.state.completed = True
            
        except Exception as e:
            self.state.error = str(e)
            logger.error(format_agent_message("data_generator", f"Error: {str(e)}"))
            
        return self.state
    
    def _load_data(self) -> None:
        """Load input data from files."""
        train_file = self.data_config.get("train_file")
        val_file = self.data_config.get("validation_file")
        
        if not train_file:
            raise AgentException("No training data file specified")
            
        try:
            train_data = self._read_data_file(train_file)
            val_data = self._read_data_file(val_file) if val_file else None
            
            self.state.input_data = {
                "train": train_data,
                "validation": val_data
            }
            
        except Exception as e:
            raise AgentException(f"Error loading data: {str(e)}")
    
    def _read_data_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read data from a file (supports JSON and CSV)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AgentException(f"File not found: {file_path}")
            
        if file_path.suffix == ".json":
            with open(file_path) as f:
                return json.load(f)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            return df.to_dict("records")
        else:
            raise AgentException(f"Unsupported file format: {file_path.suffix}")
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic training data."""
        logger.info(format_agent_message("data_generator", "Generating synthetic data"))
        
        gen_config = self.config.get("data_generation", {})
        num_samples = gen_config.get("num_samples", 100)
        
        # Example synthetic data generation logic
        # In a real implementation, this would use more sophisticated techniques
        synthetic_data = []
        
        if self.state.input_data and self.state.input_data["train"]:
            # Use input data as templates for generation
            template_data = self.state.input_data["train"]
            
            # TODO: Implement actual synthetic data generation
            # This could involve:
            # 1. Using templates from input data
            # 2. Applying transformations
            # 3. Using LLM to generate variations
            # 4. Ensuring data quality and diversity
            
            self.state.generated_data = {
                "synthetic": synthetic_data
            }
    
    def _prepare_dataset(self) -> None:
        """Prepare the final dataset for training."""
        if not self.state.input_data:
            raise AgentException("No input data available")
            
        # Combine original and synthetic data if available
        train_data = self.state.input_data["train"]
        if self.state.generated_data and self.state.generated_data.get("synthetic"):
            train_data.extend(self.state.generated_data["synthetic"])
            
        # Convert to HuggingFace datasets
        datasets = {
            "train": Dataset.from_dict({
                k: [d[k] for d in train_data]
                for k in train_data[0].keys()
            })
        }
        
        if self.state.input_data.get("validation"):
            val_data = self.state.input_data["validation"]
            datasets["validation"] = Dataset.from_dict({
                k: [d[k] for d in val_data]
                for k in val_data[0].keys()
            })
            
        self.state.dataset = DatasetDict(datasets)
    
    def _generate_statistics(self) -> None:
        """Generate statistics about the dataset."""
        if not self.state.dataset:
            return
            
        stats = {}
        
        for split, dataset in self.state.dataset.items():
            split_stats = {
                "num_examples": len(dataset),
                "columns": list(dataset.features.keys()),
            }
            
            # Calculate text statistics if applicable
            text_column = self.data_config.get("text_column")
            if text_column and text_column in dataset.features:
                samples = dataset[text_column]
                split_stats.update({
                    "avg_length": sum(len(s.split()) for s in samples) / len(samples),
                    "max_length": max(len(s.split()) for s in samples),
                    "min_length": min(len(s.split()) for s in samples)
                })
                
            stats[split] = split_stats
            
        self.state.data_stats = stats
