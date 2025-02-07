"""Trainer agent for managing the model fine-tuning process."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path

from ..trainer import LLMTrainer, TrainingConfig
from .utils import AgentState, AgentException, format_agent_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainerAgentState(AgentState):
    """State for the trainer agent."""
    model_path: Optional[str] = None
    training_stats: Dict[str, Any] = field(default_factory=dict)
    checkpoint_paths: List[str] = field(default_factory=list)
    current_epoch: int = 0
    current_step: int = 0
    best_model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

class TrainerAgent:
    """Agent responsible for managing model fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = TrainerAgentState(config=config)
        self.trainer = None
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})
        
    def run(self, state: Optional[TrainerAgentState] = None) -> TrainerAgentState:
        """Run the training process."""
        if state:
            self.state = state
            
        try:
            # Initialize trainer
            self._initialize_trainer()
            
            # Start training
            self._train()
            
            # Save final model
            self._save_model()
            
            self.state.completed = True
            
        except Exception as e:
            self.state.error = str(e)
            logger.error(format_agent_message("trainer", f"Error: {str(e)}"))
            
        return self.state
    
    def _initialize_trainer(self) -> None:
        """Initialize the LLM trainer."""
        logger.info(format_agent_message("trainer", "Initializing trainer"))
        
        # Validate model configuration
        if not self.model_config.get("base_model_name"):
            raise AgentException("No base model specified")
            
        # Setup output directory
        output_dir = self.training_config.get("output_dir", "outputs")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create training configuration
        training_config = TrainingConfig(
            model_name=self.model_config["base_model_name"],
            output_dir=output_dir,
            train_dataset=self.state.config.get("train_dataset"),
            eval_dataset=self.state.config.get("eval_dataset"),
            model_config=self.model_config,
            training_config=self.training_config,
            use_peft=self.training_config.get("use_peft", True),
            use_rlhf=self.training_config.get("use_rlhf", False)
        )
        
        self.trainer = LLMTrainer(training_config)
        self.state.model_path = output_dir
        
    def _train(self) -> None:
        """Run the training process."""
        if not self.trainer:
            raise AgentException("Trainer not initialized")
            
        logger.info(format_agent_message("trainer", "Starting training"))
        
        # Run training
        try:
            result = self.trainer.train()
            self._process_training_result(result)
        except Exception as e:
            raise AgentException(f"Training failed: {str(e)}")
    
    def _process_training_result(self, result: Dict[str, Any]) -> None:
        """Process and store training results."""
        # Extract training statistics
        self.state.training_stats = {
            "train_runtime": result.get("train_runtime", 0),
            "train_samples_per_second": result.get("train_samples_per_second", 0),
            "train_steps_per_second": result.get("train_steps_per_second", 0),
            "total_steps": result.get("total_steps", 0),
        }
        
        # Extract metrics
        if "metrics" in result:
            self.state.metrics = result["metrics"]
            
        # Track best model
        if "best_model_checkpoint" in result:
            self.state.best_model_path = result["best_model_checkpoint"]
            
        # Save checkpoints info
        if "checkpoints" in result:
            self.state.checkpoint_paths = result["checkpoints"]
    
    def _save_model(self) -> None:
        """Save the final model and training artifacts."""
        if not self.trainer:
            return
            
        logger.info(format_agent_message("trainer", "Saving model"))
        
        try:
            # Save model and tokenizer
            self.trainer.save()
            
            # Save training configuration
            config_path = os.path.join(self.state.model_path, "training_config.json")
            Path(config_path).write_text(str(self.config))
            
            # Save metrics
            metrics_path = os.path.join(self.state.model_path, "metrics.json")
            Path(metrics_path).write_text(str(self.state.metrics))
            
        except Exception as e:
            logger.error(format_agent_message(
                "trainer",
                f"Error saving model: {str(e)}"
            ))
            
    def get_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return {
            "current_epoch": self.state.current_epoch,
            "current_step": self.state.current_step,
            "metrics": self.state.metrics,
            "checkpoint_paths": self.state.checkpoint_paths,
            "best_model_path": self.state.best_model_path
        }
