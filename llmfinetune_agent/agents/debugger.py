"""Debugger agent for analyzing and improving model performance."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
from pathlib import Path

from .utils import AgentState, AgentException, format_agent_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DebuggerAnalysis:
    """Container for debugging analysis results."""
    issues: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    priority: str  # high, medium, low

@dataclass
class DebuggerState(AgentState):
    """State for the debugger agent."""
    evaluation_results: Optional[Dict[str, Any]] = None
    analysis_results: List[DebuggerAnalysis] = field(default_factory=list)
    current_analysis: Optional[str] = None
    suggestions: List[Dict[str, Any]] = field(default_factory=list)

class DebuggerAgent:
    """Agent responsible for debugging and improving model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = DebuggerState(config=config)
        self.debug_config = config.get("debugging", {})
        self.thresholds = config.get("evaluation", {}).get("thresholds", {})
        
    def run(self, state: Optional[DebuggerState] = None) -> DebuggerState:
        """Run the debugging process."""
        if state:
            self.state = state
            
        try:
            # Analyze evaluation results
            self._analyze_results()
            
            # Generate improvement suggestions
            self._generate_suggestions()
            
            # Create debugging report
            self._create_report()
            
            self.state.completed = True
            
        except Exception as e:
            self.state.error = str(e)
            logger.error(format_agent_message("debugger", f"Error: {str(e)}"))
            
        return self.state
    
    def _analyze_results(self) -> None:
        """Analyze evaluation results for issues."""
        if not self.state.evaluation_results:
            raise AgentException("No evaluation results provided")
            
        analysis_results = []
        
        # Analyze metrics
        metrics_analysis = self._analyze_metrics()
        if metrics_analysis:
            analysis_results.append(metrics_analysis)
            
        # Analyze error patterns
        error_analysis = self._analyze_error_patterns()
        if error_analysis:
            analysis_results.append(error_analysis)
            
        # Analyze training statistics
        training_analysis = self._analyze_training_stats()
        if training_analysis:
            analysis_results.append(training_analysis)
            
        self.state.analysis_results = analysis_results
    
    def _analyze_metrics(self) -> Optional[DebuggerAnalysis]:
        """Analyze model metrics against thresholds."""
        metrics = self.state.evaluation_results.get("metrics", {})
        issues = []
        recommendations = []
        
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric)
            if threshold and value < threshold:
                issues.append({
                    "type": "metric_below_threshold",
                    "metric": metric,
                    "value": value,
                    "threshold": threshold
                })
                
                recommendations.append({
                    "type": "improve_metric",
                    "metric": metric,
                    "suggestion": self._get_metric_improvement_suggestion(metric)
                })
                
        if issues:
            return DebuggerAnalysis(
                issues=issues,
                recommendations=recommendations,
                priority="high" if any(i["value"] < i["threshold"] * 0.8 for i in issues) else "medium"
            )
        return None
    
    def _analyze_error_patterns(self) -> Optional[DebuggerAnalysis]:
        """Analyze patterns in model errors."""
        error_analysis = self.state.evaluation_results.get("error_analysis", {})
        length_analysis = error_analysis.get("length_analysis", {})
        worst_examples = error_analysis.get("worst_examples", [])
        
        issues = []
        recommendations = []
        
        # Check for length discrepancies
        if length_analysis:
            avg_diff = length_analysis.get("length_diff", 0)
            if avg_diff > self.debug_config.get("max_length_diff", 5):
                issues.append({
                    "type": "length_mismatch",
                    "avg_difference": avg_diff
                })
                recommendations.append({
                    "type": "length_control",
                    "suggestion": "Consider adding length control to generation or adjusting max_length"
                })
                
        # Analyze worst examples for patterns
        if worst_examples:
            error_patterns = self._identify_error_patterns(worst_examples)
            if error_patterns:
                issues.extend(error_patterns["issues"])
                recommendations.extend(error_patterns["recommendations"])
                
        if issues:
            return DebuggerAnalysis(
                issues=issues,
                recommendations=recommendations,
                priority="medium"
            )
        return None
    
    def _analyze_training_stats(self) -> Optional[DebuggerAnalysis]:
        """Analyze training statistics for issues."""
        training_stats = self.state.evaluation_results.get("training_stats", {})
        issues = []
        recommendations = []
        
        # Check for learning rate issues
        if "learning_rate" in training_stats:
            lr = training_stats["learning_rate"]
            if lr > self.debug_config.get("max_learning_rate", 5e-5):
                issues.append({
                    "type": "high_learning_rate",
                    "value": lr
                })
                recommendations.append({
                    "type": "adjust_learning_rate",
                    "suggestion": "Consider reducing learning rate"
                })
                
        # Check for gradient issues
        if "gradient_norm" in training_stats:
            grad_norm = training_stats["gradient_norm"]
            if grad_norm > self.debug_config.get("max_grad_norm", 1.0):
                issues.append({
                    "type": "high_gradient_norm",
                    "value": grad_norm
                })
                recommendations.append({
                    "type": "gradient_control",
                    "suggestion": "Consider gradient clipping or reducing batch size"
                })
                
        if issues:
            return DebuggerAnalysis(
                issues=issues,
                recommendations=recommendations,
                priority="high" if any(i["type"] == "high_gradient_norm" for i in issues) else "medium"
            )
        return None
    
    def _get_metric_improvement_suggestion(self, metric: str) -> str:
        """Get improvement suggestion for a specific metric."""
        suggestions = {
            "rouge": "Consider fine-tuning with more diverse examples or adjusting max_length",
            "bleu": "Consider improving tokenization or adding more reference examples",
            "bertscore": "Consider additional pre-training or domain adaptation"
        }
        return suggestions.get(metric, "Consider adjusting model parameters or training data")
    
    def _identify_error_patterns(self, examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Identify patterns in worst performing examples."""
        patterns = {"issues": [], "recommendations": []}
        
        # Analyze input lengths
        input_lengths = [len(ex["input"].split()) for ex in examples]
        avg_length = sum(input_lengths) / len(input_lengths)
        
        if avg_length > self.debug_config.get("input_length_threshold", 512):
            patterns["issues"].append({
                "type": "long_inputs",
                "avg_length": avg_length
            })
            patterns["recommendations"].append({
                "type": "input_length",
                "suggestion": "Consider chunking long inputs or increasing model context window"
            })
            
        # TODO: Add more pattern analysis:
        # - Common failure modes
        # - Specific token/word issues
        # - Domain-specific problems
        
        return patterns
    
    def _generate_suggestions(self) -> None:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        for analysis in self.state.analysis_results:
            # Prioritize high-priority issues
            if analysis.priority == "high":
                suggestions.extend([
                    {**r, "priority": "high"}
                    for r in analysis.recommendations
                ])
            else:
                suggestions.extend([
                    {**r, "priority": analysis.priority}
                    for r in analysis.recommendations
                ])
                
        # Sort suggestions by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self.state.suggestions = sorted(
            suggestions,
            key=lambda x: priority_order[x["priority"]]
        )
    
    def _create_report(self) -> None:
        """Create debugging report."""
        if not self.state.analysis_results:
            return
            
        report_path = Path(self.config.get("output_dir", "outputs")) / "debug_report.json"
        report = {
            "analysis_results": [
                {
                    "issues": a.issues,
                    "recommendations": a.recommendations,
                    "priority": a.priority
                }
                for a in self.state.analysis_results
            ],
            "suggestions": self.state.suggestions
        }
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            import json
            json.dump(report, f, indent=2)
            
    def get_suggestions(self) -> List[Dict[str, Any]]:
        """Get improvement suggestions."""
        return self.state.suggestions
