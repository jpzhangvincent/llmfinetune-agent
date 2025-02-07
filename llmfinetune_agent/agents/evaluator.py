"""Evaluator agent for assessing model performance."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import AgentState, AgentException, format_agent_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    predictions: List[str]
    references: List[str]
    examples: List[Dict[str, Any]]
    analysis: Dict[str, Any]

@dataclass
class EvaluatorState(AgentState):
    """State for the evaluator agent."""
    model_path: Optional[str] = None
    evaluation_results: Optional[EvaluationResult] = None
    current_metric: Optional[str] = None
    metrics_results: Dict[str, float] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)

class EvaluatorAgent:
    """Agent responsible for evaluating model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = EvaluatorState(config=config)
        self.eval_config = config.get("evaluation", {})
        self.model = None
        self.tokenizer = None
        
    def run(self, state: Optional[EvaluatorState] = None) -> EvaluatorState:
        """Run the evaluation process."""
        if state:
            self.state = state
            
        try:
            # Load model and tokenizer
            self._load_model()
            
            # Run evaluation
            self._evaluate()
            
            # Perform error analysis
            self._analyze_errors()
            
            # Generate evaluation report
            self._generate_report()
            
            self.state.completed = True
            
        except Exception as e:
            self.state.error = str(e)
            logger.error(format_agent_message("evaluator", f"Error: {str(e)}"))
            
        return self.state
    
    def _load_model(self) -> None:
        """Load the fine-tuned model and tokenizer."""
        model_path = self.state.model_path
        if not model_path or not Path(model_path).exists():
            raise AgentException("Model path not found")
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise AgentException(f"Error loading model: {str(e)}")
    
    def _evaluate(self) -> None:
        """Run evaluation using configured metrics."""
        metrics = self.eval_config.get("metrics", ["rouge", "bleu", "bertscore"])
        eval_dataset = self.state.config.get("eval_dataset")
        
        if not eval_dataset:
            raise AgentException("No evaluation dataset provided")
            
        results = {}
        predictions = []
        references = []
        
        # Run generation on evaluation dataset
        for example in eval_dataset:
            input_text = example.get("input", "")
            reference = example.get("output", "")
            
            # Generate prediction
            inputs = self.tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"].to(self.model.device),
                    max_new_tokens=self.eval_config.get("max_new_tokens", 128),
                    num_return_sequences=1
                )
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
            
        # Calculate metrics
        for metric_name in metrics:
            try:
                metric = evaluate.load(metric_name)
                score = metric.compute(
                    predictions=predictions,
                    references=references
                )
                results[metric_name] = score
            except Exception as e:
                logger.warning(format_agent_message(
                    "evaluator",
                    f"Error computing {metric_name}: {str(e)}"
                ))
                
        self.state.metrics_results = results
        
        # Store examples for analysis
        examples = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            examples.append({
                "id": i,
                "input": eval_dataset[i].get("input", ""),
                "prediction": pred,
                "reference": ref
            })
            
        self.state.evaluation_results = EvaluationResult(
            metrics=results,
            predictions=predictions,
            references=references,
            examples=examples,
            analysis={}
        )
    
    def _analyze_errors(self) -> None:
        """Perform error analysis on evaluation results."""
        if not self.state.evaluation_results:
            return
            
        analysis = {}
        
        # Analyze prediction lengths
        pred_lengths = [len(p.split()) for p in self.state.evaluation_results.predictions]
        ref_lengths = [len(r.split()) for r in self.state.evaluation_results.references]
        
        analysis["length_analysis"] = {
            "avg_pred_length": sum(pred_lengths) / len(pred_lengths),
            "avg_ref_length": sum(ref_lengths) / len(ref_lengths),
            "length_diff": sum(abs(p - r) for p, r in zip(pred_lengths, ref_lengths)) / len(pred_lengths)
        }
        
        # Find worst performing examples
        examples = self.state.evaluation_results.examples
        if "rouge" in self.state.metrics_results:
            rouge_scores = self.state.metrics_results["rouge"]
            sorted_examples = sorted(
                examples,
                key=lambda x: rouge_scores[x["id"]]["rouge1"]
            )
            analysis["worst_examples"] = sorted_examples[:5]
            
        self.state.error_analysis = analysis
    
    def _generate_report(self) -> None:
        """Generate evaluation report."""
        if not self.state.evaluation_results or not self.state.error_analysis:
            return
            
        report_path = Path(self.state.model_path) / "evaluation_report.json"
        
        report = {
            "metrics": self.state.metrics_results,
            "error_analysis": self.state.error_analysis,
            "examples": {
                "best": self.state.evaluation_results.examples[-5:],
                "worst": self.state.evaluation_results.examples[:5]
            }
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
    def get_results(self) -> Dict[str, Any]:
        """Get evaluation results."""
        if not self.state.evaluation_results:
            return {}
            
        return {
            "metrics": self.state.metrics_results,
            "error_analysis": self.state.error_analysis
        }
