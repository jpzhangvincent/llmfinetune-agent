"""Orchestrator agent for coordinating the fine-tuning workflow."""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, AsyncIterator

from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .utils import AgentException, format_agent_message
from .data_generator import DataGenerator
from .trainer_agent import TrainerAgent 
from .evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStage(str, Enum):
    """Stages in the fine-tuning workflow."""
    DATA_GENERATION = "data_generation"
    BASELINE_EVAL = "baseline_eval"
    TRAINING = "training"
    FINAL_EVAL = "final_eval"
    COMPLETED = "completed"

class AgentState(TypedDict):
    """State maintained throughout the workflow."""
    messages: List[str]
    current_stage: WorkflowStage
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]

class Orchestrator:
    """Agent responsible for orchestrating the fine-tuning workflow."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.data_generator = DataGenerator(config)
        self.trainer = TrainerAgent(config)
        self.evaluator = Evaluator(config)
        
        # Initialize LLM for supervision
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "gpt-3.5-turbo"),
            temperature=0
        )
        
    def create_workflow(self) -> StateGraph:
        """Create the workflow graph following the Bespoke Labs example pattern."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each stage
        workflow.add_node("data_generation", self.handle_data_generation)
        workflow.add_node("baseline_eval", self.handle_baseline_eval)
        workflow.add_node("training", self.handle_training)
        workflow.add_node("final_eval", self.handle_final_eval)
        workflow.add_node("supervisor", self.handle_supervision)
        
        # Define edges
        workflow.add_edge("supervisor", "data_generation")
        workflow.add_edge("data_generation", "supervisor")
        workflow.add_edge("supervisor", "baseline_eval")
        workflow.add_edge("baseline_eval", "supervisor")
        workflow.add_edge("supervisor", "training")
        workflow.add_edge("training", "supervisor")
        workflow.add_edge("supervisor", "final_eval")
        workflow.add_edge("final_eval", "supervisor")
        workflow.add_edge("supervisor", END)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        return workflow.compile()
    
    async def determine_next_stage(self, state: Dict[str, Any]) -> str:
        """Determine the next stage in the workflow."""
        current_stage = state.get("current_stage", WorkflowStage.DATA_GENERATION)
        
        # Define stage transitions
        transitions = {
            WorkflowStage.DATA_GENERATION: "data_generator",
            WorkflowStage.BASELINE_EVAL: "evaluator",
            WorkflowStage.TRAINING: "trainer",
            WorkflowStage.FINAL_EVAL: "evaluator",
            WorkflowStage.COMPLETED: "end"
        }
        
        return transitions.get(current_stage, "end")

    def handle_supervision(self, state: AgentState) -> Dict:
        """Supervisor node that coordinates between stages."""
        current_stage = state.get("current_stage", WorkflowStage.DATA_GENERATION)
        
        # Define stage transitions
        transitions = {
            WorkflowStage.DATA_GENERATION: WorkflowStage.BASELINE_EVAL,
            WorkflowStage.BASELINE_EVAL: WorkflowStage.TRAINING,
            WorkflowStage.TRAINING: WorkflowStage.FINAL_EVAL,
            WorkflowStage.FINAL_EVAL: WorkflowStage.COMPLETED
        }
        
        next_stage = transitions.get(current_stage)
        if next_stage == WorkflowStage.COMPLETED:
            return {
                "config": self.config,
                "messages": ["Workflow completed"],
                "current_stage": WorkflowStage.COMPLETED,
                "completed": True,
                "artifacts": state.get("artifacts", {}),
                "metrics": state.get("metrics", {})
            }
            
        return {
            "config": self.config,
            "messages": [f"Moving to stage: {next_stage}"],
            "current_stage": next_stage,
            "artifacts": state.get("artifacts", {}),
            "metrics": state.get("metrics", {})
        }
    
    def handle_data_generation(self, state: AgentState) -> Dict:
        """Handle data generation stage."""
        self.logger.info("Generating training data...")
        
        try:
            # Generate data following Bespoke Labs example
            train_data = self.data_generator.generate_data()
            
            return {
                "artifacts": {"train_data": train_data},
                "messages": ["Data generation complete"],
                "current_stage": WorkflowStage.BASELINE_EVAL
            }
        except Exception as e:
            self.logger.error(f"Data generation failed: {str(e)}")
            raise
    
    def handle_baseline_eval(self, state: AgentState) -> Dict:
        """Handle baseline evaluation stage."""
        self.logger.info("Evaluating baseline model...")
        
        try:
            # Get training data
            train_data = state["artifacts"]["train_data"]
            
            # Evaluate baseline model
            baseline_metrics = self.evaluator.evaluate(None, train_data)
            
            return {
                "artifacts": {**state["artifacts"]},
                "metrics": baseline_metrics,
                "messages": [f"Baseline evaluation complete: {baseline_metrics}"],
                "current_stage": WorkflowStage.TRAINING
            }
        except Exception as e:
            self.logger.error(f"Baseline evaluation failed: {str(e)}")
            raise
            
    def handle_training(self, state: AgentState) -> Dict:
        """Handle model training stage."""
        self.logger.info("Training model...")
        
        try:
            # Get training data
            train_data = state["artifacts"]["train_data"]
            
            # Train model
            trained_model = self.trainer.train(train_data)
            
            return {
                "artifacts": {
                    **state["artifacts"],
                    "trained_model": trained_model
                },
                "messages": ["Training complete"],
                "current_stage": WorkflowStage.FINAL_EVAL
            }
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        
    def handle_final_eval(self, state: AgentState) -> Dict:
        """Handle final evaluation stage."""
        self.logger.info("Evaluating fine-tuned model...")
        
        try:
            # Get model and test data
            model = state["artifacts"]["trained_model"]
            test_data = state["artifacts"]["train_data"]  # Using same data for demo
            
            # Evaluate fine-tuned model
            final_metrics = self.evaluator.evaluate(model, test_data)
            
            return {
                "artifacts": {**state["artifacts"]},
                "metrics": final_metrics,
                "messages": [f"Final evaluation complete: {final_metrics}"],
                "current_stage": WorkflowStage.COMPLETED
            }
        except Exception as e:
            self.logger.error(f"Final evaluation failed: {str(e)}")
            raise
        
    async def arun(self, state: Optional[AgentState] = None) -> AsyncIterator[Dict[str, Any]]:
        """Run the orchestrator workflow asynchronously."""
        self.logger.info("Starting orchestrator workflow...")
        
        try:
            # Create workflow
            workflow = self.create_workflow()
            
            # Initialize state if not provided
            if state is None:
                state = {
                    "messages": [],
                    "current_stage": WorkflowStage.DATA_GENERATION,
                    "artifacts": {},
                    "metrics": {}
                }
            
            # Execute workflow
            self.logger.info(f"Current stage: {state.get('current_stage', WorkflowStage.DATA_GENERATION)}")
            result = await workflow.ainvoke(state)
            
            # Log results
            self.logger.info("Workflow completed successfully")
            if result.get('metrics'):
                self.logger.info(f"Final evaluation results: {result['metrics']}")
            
            yield result
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            yield {
                "error": str(e),
                "current_stage": state.get("current_stage", WorkflowStage.DATA_GENERATION)
            }
