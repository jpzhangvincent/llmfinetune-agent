"""Tests for the evaluator agent."""
import json
import pytest
import torch
from unittest.mock import Mock, patch
from pathlib import Path

from llmfinetune_agent.agents.evaluator import (
    EvaluatorAgent,
    EvaluatorState,
    EvaluationResult
)

@pytest.fixture
def mock_config():
    return {
        "evaluation": {
            "metrics": ["precision", "recall", "f1"],
            "similarity_threshold": 0.70,
            "max_new_tokens": 128
        },
        "model": {
            "base_model_name": "test/model"
        }
    }

@pytest.fixture
def mock_eval_dataset():
    return [
        {
            "name": "Test Product 1",
            "description": "A product with feature1 and feature2.",
            "features": ["feature1", "feature2"]
        },
        {
            "name": "Test Product 2", 
            "description": "A product with feature3 and feature4.",
            "features": ["feature3", "feature4"]
        }
    ]

@pytest.fixture
def mock_model():
    model = Mock()
    model.device = "cpu"
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.decode.return_value = "Test prediction"
    return tokenizer

@pytest.fixture
def evaluator_agent(mock_config):
    with patch("transformers.AutoModelForCausalLM"), \
         patch("transformers.AutoTokenizer"):
        agent = EvaluatorAgent(mock_config)
        return agent

def test_evaluator_initialization(evaluator_agent, mock_config):
    """Test evaluator agent initialization."""
    assert evaluator_agent.config == mock_config
    assert isinstance(evaluator_agent.state, EvaluatorState)
    assert evaluator_agent.eval_config == mock_config["evaluation"]

def test_evaluator_run_without_model_path(evaluator_agent):
    """Test evaluator run fails without model path."""
    state = evaluator_agent.run()
    assert state.error is not None
    assert "Model path not found" in state.error

def test_evaluator_run_with_invalid_model_path(evaluator_agent):
    """Test evaluator run fails with invalid model path."""
    evaluator_agent.state.model_path = "invalid/path"
    state = evaluator_agent.run()
    assert state.error is not None
    assert "Model path not found" in state.error

def test_feature_evaluation(evaluator_agent):
    """Test feature extraction evaluation."""
    predictions = [["feature1", "feature2"], ["feature3", "feature4"]]
    references = [["feature1", "feature2"], ["feature3", "feature4"]]
    
    metrics = evaluator_agent._evaluate_features(predictions, references)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

def test_feature_evaluation_partial_match(evaluator_agent):
    """Test feature extraction evaluation with partial matches."""
    predictions = [["feature1"], ["feature3", "feature4", "extra"]]
    references = [["feature1", "feature2"], ["feature3", "feature4"]]
    
    metrics = evaluator_agent._evaluate_features(predictions, references)
    
    assert metrics["precision"] < 1.0
    assert metrics["recall"] < 1.0
    assert metrics["f1"] < 1.0

def test_feature_evaluation_fuzzy_match(evaluator_agent):
    """Test feature extraction evaluation with fuzzy matching."""
    predictions = [["feature one", "feature two"], ["feature three", "feature four"]]
    references = [["feature1", "feature2"], ["feature3", "feature4"]]
    
    metrics = evaluator_agent._evaluate_features(predictions, references)
    
    # With fuzzy matching, similar features should still get some credit
    assert metrics["precision"] > 0.0
    assert metrics["recall"] > 0.0
    assert metrics["f1"] > 0.0

@pytest.mark.asyncio
async def test_evaluator_langraph_node():
    """Test evaluator agent as a langraph node."""
    from langgraph.graph import Graph, START, END
    
    # Mock evaluator config
    config = {
        "evaluation": {
            "metrics": ["precision", "recall", "f1"],
            "similarity_threshold": 0.85
        }
    }
    
    # Create evaluator node
    async def evaluator_node(state):
        agent = EvaluatorAgent(config)
        agent.state.model_path = state.get("model_path")
        agent.state.config["eval_dataset"] = state.get("eval_dataset")
        result = agent.run()
        return {
            "metrics": result.metrics_results,
            "error_analysis": result.error_analysis
        }
    
    # Create graph
    workflow = Graph()
    workflow.add_node("evaluator", evaluator_node)
    
    # Add edges
    workflow.set_entry_point("evaluator")
    workflow.add_edge("evaluator", END)
    
    # Test graph execution
    state = {
        "model_path": "test/model",
        "eval_dataset": [
            {
                "name": "Test Product",
                "description": "A product with feature1.",
                "features": ["feature1"]
            }
        ]
    }
    
    compiled_workflow = workflow.compile()
    result = await compiled_workflow.ainvoke(state)
    assert "metrics" in result
    assert "error_analysis" in result

def test_evaluation_result_creation():
    """Test creation of evaluation results."""
    result = EvaluationResult(
        metrics={"precision": 0.8, "recall": 0.9, "f1": 0.85},
        predictions=[["feature1"], ["feature3"]],
        references=[["feature1", "feature2"], ["feature3", "feature4"]],
        examples=[
            {
                "id": 0,
                "name": "Test Product 1",
                "description": "Test description 1",
                "predicted_features": ["feature1"],
                "true_features": ["feature1", "feature2"]
            },
            {
                "id": 1,
                "name": "Test Product 2",
                "description": "Test description 2",
                "predicted_features": ["feature3"],
                "true_features": ["feature3", "feature4"]
            }
        ],
        analysis={"feature_count_analysis": {"avg_pred_features": 1.0}}
    )
    
    assert result.metrics["precision"] == 0.8
    assert result.metrics["recall"] == 0.9
    assert result.metrics["f1"] == 0.85
    assert len(result.predictions) == 2
    assert len(result.examples) == 2
    assert "feature_count_analysis" in result.analysis

def test_evaluator_error_analysis(evaluator_agent, mock_eval_dataset):
    """Test error analysis functionality."""
    evaluator_agent.state.evaluation_results = EvaluationResult(
        metrics={"precision": 0.8, "recall": 0.9, "f1": 0.85},
        predictions=[["feature1"], ["feature3"]],
        references=[["feature1", "feature2"], ["feature3", "feature4"]],
        examples=[
            {
                "id": 0,
                "name": "Test Product 1",
                "description": "Test description 1",
                "predicted_features": ["feature1"],
                "true_features": ["feature1", "feature2"]
            },
            {
                "id": 1,
                "name": "Test Product 2",
                "description": "Test description 2",
                "predicted_features": ["feature3"],
                "true_features": ["feature3", "feature4"]
            }
        ],
        analysis={}
    )
    
    evaluator_agent._analyze_errors()
    
    assert "feature_count_analysis" in evaluator_agent.state.error_analysis
    assert "worst_examples" in evaluator_agent.state.error_analysis
    assert evaluator_agent.state.error_analysis["feature_count_analysis"]["avg_pred_features"] == 1.0

def test_evaluator_report_generation(evaluator_agent, tmp_path):
    """Test evaluation report generation."""
    evaluator_agent.state.model_path = str(tmp_path)
    evaluator_agent.state.metrics_results = {"precision": 0.8, "recall": 0.9, "f1": 0.85}
    evaluator_agent.state.error_analysis = {"feature_count_analysis": {"avg_pred_features": 1.0}}
    evaluator_agent.state.evaluation_results = EvaluationResult(
        metrics={"precision": 0.8, "recall": 0.9, "f1": 0.85},
        predictions=[["feature1"]],
        references=[["feature1", "feature2"]],
        examples=[{
            "id": 0,
            "name": "Test Product",
            "description": "Test description",
            "predicted_features": ["feature1"],
            "true_features": ["feature1", "feature2"]
        }],
        analysis={}
    )
    
    evaluator_agent._generate_report()
    
    report_path = tmp_path / "evaluation_report.json"
    assert report_path.exists()
    
    with open(report_path) as f:
        report = json.load(f)
        assert "metrics" in report
        assert "error_analysis" in report
        assert "examples" in report
