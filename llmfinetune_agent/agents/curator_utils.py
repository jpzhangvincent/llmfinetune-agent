"""Utilities for working with bespokelabs-curator."""
import logging
from typing import Any, Dict, List

from bespokelabs import curator
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ProductFeatures(BaseModel):
    """Model for product features response."""
    features: List[str]

class Curator(curator.LLM):
    """Custom Curator class for generating and evaluating data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize curator with config."""
        self.config = config
        model_name = config.get("model", {}).get("base_model_name", "ollama/llama3.1:8b")
        backend_params = {
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 30000000
        }
        if "ollama" in model_name:
            backend_params.update({
                "base_url": "http://localhost:11434",
            })
        super().__init__(model_name=model_name, backend_params=backend_params)

    def prompt(self, row: Dict[str, Any]) -> str:
        """Generate prompt for feature extraction."""
        return f"""You are given a product's name, description and features. You will generate a list of features for the product.

        An example input is:
        {{
            "name": "Apple AirPods Pro",
            "description": "The Apple AirPods Pro are a pair of wireless earbuds that are designed for comfort and convenience. They are lightweight in-ear earbuds and contoured for a comfortable fit, and they sit at an angle for easy access to the controls. The AirPods Pro also have a stem that is 33% shorter than the second generation AirPods, which makes them more compact and easier to store. The AirPods Pro also have a force sensor to easily control music and calls, and they have Spatial Audio with dynamic head tracking, which provides an immersive, three-dimensional listening experience.",
        }}

        An example output is:
        {{
            "features": [
                "lightweight in-ear earbuds",
                "contoured for a comfortable fit",
                "sit at an angle for easy access to the controls",
                "stem is 33% shorter than the second generation AirPods",
                "force sensor to easily control music and calls",
                "Spatial Audio with dynamic head tracking",
                "immersive, three-dimensional listening experience"
            ]
        }}

        Now, generate a list of features for the product. You should output all the features that are mentioned in the description exactly as they are written. You should not miss any features, or add any features that are not mentioned in the description.

        Your output should be in this format.
        ```json{{"features": ["feature 1","feature 2","feature 3",...]}}```

        Product:
          Name: {row.get('product_name', '')}
          Description: {row.get('description', '')}
        Output:
        """

    def parse(self, row: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """Parse the LLM response to extract features."""
        try:
            if isinstance(response, str):
                # Extract JSON from markdown code block if present
                import re
                pattern = r"```json(.*?)```"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    import json
                    response = json.loads(json_str)
            
            # Convert to Pydantic model for validation
            features = ProductFeatures(**response)
            
            # Update row with extracted features
            row['features'] = features.features
            return row
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            row['features'] = []
            return row

class EvaluationCurator(curator.LLM):
    """Curator class for evaluating feature extraction."""

    def prompt(self, row: Dict[str, Any]) -> str:
        """Generate evaluation prompt."""
        return f"""Evaluate the quality of feature extraction for this product.

        Product Name: {row.get('product_name', '')}
        Description: {row.get('description', '')}

        Extracted Features:
        {row.get('features', [])}

        True Features:
        {row.get('true_features', [])}

        Evaluate the following:
        1. Precision: What percentage of extracted features are correct?
        2. Recall: What percentage of true features were found?
        3. Accuracy: Overall accuracy of feature extraction

        Output your evaluation as JSON:
        ```json
        {{
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "comments": "explanation"
        }}
        ```
        """

    def parse(self, row: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """Parse evaluation response."""
        try:
            if isinstance(response, str):
                import re
                pattern = r"```json(.*?)```"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    import json
                    metrics = json.loads(json_str)
                    row.update(metrics)
            return row
        except Exception as e:
            logger.error(f"Error parsing evaluation: {e}")
            return row

def create_curator(config: Dict[str, Any], evaluation: bool = False) -> curator.LLM:
    """Factory function to create appropriate curator instance."""
    if evaluation:
        return EvaluationCurator(config)
    return Curator(config)
