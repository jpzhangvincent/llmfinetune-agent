"""Orchestrator agent for coordinating the fine-tuning workflow."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor

from .utils import AgentState, AgentException, format_agent_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """Stages in the fine-tuning workflow."""
    INITIALIZE = "initialize"
    DATA_PREPARATION = "data_preparation"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TRAINING = "training"
    EVALUATION = "evaluation"
    DEBUGGING = "debugging"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class OrchestratorState(AgentState):
    """State for the orchestrator agent."""
    current_stage: WorkflowStage = WorkflowStage.INITIALIZE
    stage_results: Dict[str, Any] = field(default_factory=dict)
    training_config: Optional[Dict[str, Any]] = None
    requires_feedback: bool = False
    feedback_message: Optional[str] = None
    stage_history: List[WorkflowStage] = field(default_factory=list)

class OrchestratorAgent:
    """Agent responsible for orchestrating the fine-tuning workflow."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = OrchestratorState(config=config)
        
    def determine_next_stage(self, state: OrchestratorState) -> WorkflowStage:
        """Determine the next stage in the workflow."""
        if state.error:
            return WorkflowStage.ERROR
            
        current = state.current_stage
        
        # Define stage transitions
        transitions = {
            WorkflowStage.INITIALIZE: WorkflowStage.DATA_PREPARATION,
            WorkflowStage.DATA_PREPARATION: (
                WorkflowStage.KNOWLEDGE_RETRIEVAL 
                if self.config.get("knowledge_retrieval", {}).get("enabled", False)
                else WorkflowStage.TRAINING
            ),
            WorkflowStage.KNOWLEDGE_RETRIEVAL: WorkflowStage.TRAINING,
            WorkflowStage.TRAINING: WorkflowStage.EVALUATION,
            WorkflowStage.EVALUATION: (
                WorkflowStage.DEBUGGING 
                if self.needs_debugging(state)
                else WorkflowStage.COMPLETED
            ),
            WorkflowStage.DEBUGGING: WorkflowStage.TRAINING,
            WorkflowStage.COMPLETED: WorkflowStage.COMPLETED,
            WorkflowStage.ERROR: WorkflowStage.ERROR
        }
        
        return transitions.get(current, WorkflowStage.ERROR)
    
    def needs_debugging(self, state: OrchestratorState) -> bool:
        """Determine if debugging is needed based on evaluation results."""
        if not state.stage_results.get("evaluation"):
            return False
            
        eval_results = state.stage_results["evaluation"]
        # Check if any metric is below threshold
        metrics = eval_results.get("metrics", {})
        thresholds = self.config.get("evaluation", {}).get("thresholds", {})
        
        for metric, value in metrics.items():
            if metric in thresholds and value < thresholds[metric]:
                return True
                
        return False
    
    def run(self, state: Optional[OrchestratorState] = None) -> OrchestratorState:
        """Run the orchestrator workflow."""
        if state:
            self.state = state
            
        while not self.state.completed and not self.state.error:
            next_stage = self.determine_next_stage(self.state)
            
            if next_stage == self.state.current_stage:
                break
                
            self.state.stage_history.append(self.state.current_stage)
            self.state.current_stage = next_stage
            
            try:
                self._handle_stage(self.state)
            except Exception as e:
                self.state.error = str(e)
                logger.error(f"Error in stage {next_stage}: {str(e)}")
                break
                
            if self.state.requires_feedback:
                break
                
        return self.state
    
    def _handle_stage(self, state: OrchestratorState) -> None:
        """Handle the current workflow stage."""
        handlers = {
            WorkflowStage.INITIALIZE: self._handle_initialize,
            WorkflowStage.DATA_PREPARATION: self._handle_data_preparation,
            WorkflowStage.KNOWLEDGE_RETRIEVAL: self._handle_knowledge_retrieval,
            WorkflowStage.TRAINING: self._handle_training,
            WorkflowStage.EVALUATION: self._handle_evaluation,
            WorkflowStage.DEBUGGING: self._handle_debugging,
            WorkflowStage.COMPLETED: self._handle_completion,
            WorkflowStage.ERROR: self._handle_error
        }
        
        handler = handlers.get(state.current_stage)
        if handler:
            handler(state)
        else:
            raise AgentException(f"No handler for stage: {state.current_stage}")
            
    def _handle_initialize(self, state: OrchestratorState) -> None:
        """Initialize the workflow."""
        logger.info(format_agent_message("orchestrator", "Initializing workflow"))
        # Validate configuration
        required_fields = ["model", "training", "data"]
        for field in required_fields:
            if field not in self.config:
                raise AgentException(f"Missing required configuration: {field}")
                
        state.training_config = self.config.get("training", {})
        
    def _handle_data_preparation(self, state: OrchestratorState) -> None:
        """Handle data preparation stage."""
        logger.info(format_agent_message("orchestrator", "Starting data preparation"))
        # Data preparation handled by DataGenerator agent
        pass
        
    def _handle_knowledge_retrieval(self, state: OrchestratorState) -> None:
        """Handle knowledge retrieval stage."""
        logger.info(format_agent_message("orchestrator", "Starting knowledge retrieval"))
        # Knowledge retrieval handled by KnowledgeRetrieval agent
        pass
        
    def _handle_training(self, state: OrchestratorState) -> None:
        """Handle training stage."""
        logger.info(format_agent_message("orchestrator", "Starting training"))
        # Training handled by Trainer agent
        pass
        
    def _handle_evaluation(self, state: OrchestratorState) -> None:
        """Handle evaluation stage."""
        logger.info(format_agent_message("orchestrator", "Starting evaluation"))
        # Evaluation handled by Evaluator agent
        pass
        
    def _handle_debugging(self, state: OrchestratorState) -> None:
        """Handle debugging stage."""
        logger.info(format_agent_message("orchestrator", "Starting debugging"))
        # Debugging handled by Debugger agent
        pass
        
    def _handle_completion(self, state: OrchestratorState) -> None:
        """Handle workflow completion."""
        logger.info(format_agent_message("orchestrator", "Workflow completed"))
        state.completed = True
        
    def _handle_error(self, state: OrchestratorState) -> None:
        """Handle error state."""
        logger.error(format_agent_message("orchestrator", f"Workflow error: {state.error}"))
        
    def handle_feedback(self, feedback: str) -> None:
        """Handle user feedback."""
        self.state.requires_feedback = False
        self.state.feedback_message = None
        
        # Process feedback and update state accordingly
        if "continue" in feedback.lower():
            logger.info(format_agent_message("orchestrator", "Continuing workflow"))
        elif "retry" in feedback.lower():
            # Reset current stage
            if self.state.stage_history:
                self.state.current_stage = self.state.stage_history.pop()
        else:
            self.state.error = "Invalid feedback received"

def create_orchestrator_graph() -> Graph:
    """Create the orchestrator workflow graph using langgraph."""
    workflow = StateGraph(AgentState)
    
    # Define nodes for each agent
    workflow.add_node("orchestrator", OrchestratorAgent.run)
    # Add other agent nodes here
    
    # Define edges
    workflow.add_edge("orchestrator", "data_generator")
    workflow.add_edge("data_generator", "knowledge_retrieval")
    workflow.add_edge("knowledge_retrieval", "trainer")
    workflow.add_edge("trainer", "evaluator")
    workflow.add_edge("evaluator", "debugger")
    workflow.add_edge("debugger", "trainer")  # Loop back for refinement
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Compile graph
    return workflow.compile()
