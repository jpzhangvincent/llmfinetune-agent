"""Evaluator agent for assessing model performance."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator

from fuzzywuzzy import fuzz
from .utils import AgentException, format_agent_message
from .curator_utils import create_curator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """Agent responsible for evaluating model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.curator = create_curator(config)
    
    async def arun(self, state: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Run the evaluator agent asynchronously."""
        try:
            logger.info(format_agent_message("evaluator", "Starting evaluation"))
            
            # Get model and eval data from state
            model = state.get("artifacts", {}).get("trained_model") if state else None
            eval_data = state.get("artifacts", {}).get("train_data") if state else []
            
            # Generate predictions using curator
            predictions, references = self._generate_predictions(eval_data)
            
            # Calculate metrics
            metrics = self._evaluate_features(predictions, references)
            
            # Save evaluation report
            self._save_report(metrics, predictions, references, eval_data)
            
            # Update and yield state
            yield {
                "config": self.config,
                "messages": [f"Evaluation complete: {metrics}"],
                "current_stage": "evaluation",
                "artifacts": {
                    "trained_model": model,
                    "eval_data": eval_data,
                    "predictions": predictions,
                    "references": references
                },
                "metrics": metrics,
                "completed": False
            }
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(format_agent_message("evaluator", error_msg))
            yield {
                "config": self.config,
                "messages": [error_msg],
                "current_stage": "error",
                "artifacts": {},
                "metrics": {},
                "error": error_msg,
                "completed": False
            }
    
    def _evaluate_features(self, predictions: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
        """Evaluate feature extraction using fuzzy matching."""
        SIMILARITY_THRESHOLD = self.eval_config.get("similarity_threshold", 0.85)
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_features, ref_features in zip(predictions, references):
            for pred in pred_features:
                # Check if any reference feature matches this prediction
                best_match = max(
                    (fuzz.ratio(pred.lower(), ref.lower()) / 100.0 for ref in ref_features),
                    default=0
                )
                if best_match >= SIMILARITY_THRESHOLD:
                    total_tp += 1
                else:
                    total_fp += 1
            
            # Count false negatives (reference features that weren't predicted)
            for ref in ref_features:
                best_match = max(
                    (fuzz.ratio(ref.lower(), pred.lower()) / 100.0 for pred in pred_features),
                    default=0
                )
                if best_match < SIMILARITY_THRESHOLD:
                    total_fn += 1
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def _generate_predictions(self, eval_data: List[Dict[str, Any]]) -> tuple[List[List[str]], List[List[str]]]:
        """Generate predictions using curator."""
        predictions = []
        references = []
        
        for example in eval_data:
            # Get predictions using curator
            result = self.curator({
                "product": example.get("name", ""),
                "description": example.get("description", "")
            })
            
            # Extract features
            pred_features = result.get("features", [])
            true_features = example.get("features", [])
            
            predictions.append(pred_features)
            references.append(true_features)
            
        return predictions, references
    
    def _save_report(
        self,
        metrics: Dict[str, float],
        predictions: List[List[str]],
        references: List[List[str]],
        eval_data: List[Dict[str, Any]]
    ) -> None:
        """Save evaluation report."""
        report = {
            "metrics": metrics,
            "examples": []
        }
        
        # Add example-level results
        for i, (pred, ref, example) in enumerate(zip(predictions, references, eval_data)):
            example_metrics = self._evaluate_features([pred], [ref])
            report["examples"].append({
                "id": i,
                "name": example.get("name", ""),
                "description": example.get("description", ""),
                "predicted_features": pred,
                "true_features": ref,
                "metrics": example_metrics
            })
        
        # Save report
        output_dir = Path(self.config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
