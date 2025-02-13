"""Data generator agent for creating and preparing training data."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, AsyncIterator, Optional

from datasets import load_dataset
from .curator_utils import create_curator
from .utils import AgentException, format_agent_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    """Agent responsible for generating and preparing training data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get("data", {})
        self.curator = create_curator(config)
        
    async def arun(self, state: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Run the data generator agent asynchronously."""
        try:
            logger.info(format_agent_message("data_generator", "Starting data generation"))
            
            # Load personas dataset for diverse product generation
            personas = load_dataset(
                self.config.get("data_generation", {}).get("personas_dataset", "proj-persona/PersonaHub"),
                self.config.get("data_generation", {}).get("personas_split", "persona")
            )
            num_personas = self.config.get("data_generation", {}).get("num_personas", 100)
            personas = personas['train'].take(num_personas)
            
            # Generate products using curator
            logger.info(format_agent_message("data_generator", "Generating products from personas"))
            generated_data = self.curator(personas)
            
            # Process generated data into expected format
            processed_data = []
            for item in generated_data:
                # Extract features from description with <feature> tags
                import re
                pattern = r"<feature>(.*?)</feature>"
                matches = re.findall(pattern, item.get('description', ''))
                
                # Clean description by removing tags
                clean_description = item.get('description', '').replace('<feature>', '').replace('</feature>', '')
                
                processed_data.append({
                    "name": item.get('product', ''),
                    "description": clean_description,
                    "features": matches if matches else item.get('features', [])
                })
            
            # Save the generated data
            output_path = self.config.get("data", {}).get("train_file", "data/train.json")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            logger.info(format_agent_message(
                "data_generator",
                f"Generated and saved {len(processed_data)} examples to {output_path}"
            ))
            
            # Update and yield state
            yield {
                "config": self.config,
                "messages": [f"Generated {len(processed_data)} training examples"],
                "current_stage": "data_generation",
                "artifacts": {
                    "train_data": processed_data,
                    "train_file": output_path
                },
                "metrics": {},
                "completed": False
            }
            
        except Exception as e:
            error_msg = f"Data generation failed: {str(e)}"
            logger.error(format_agent_message("data_generator", error_msg))
            yield {
                "config": self.config,
                "messages": [error_msg],
                "current_stage": "error",
                "artifacts": {},
                "metrics": {},
                "error": error_msg,
                "completed": False
            }
