"""Trainer agent for managing the model fine-tuning process."""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator

from .trainer_utils import LLMTrainer, TrainingConfig
from .utils import AgentException, AgentState, format_agent_message

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
    training_config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None

class TrainerAgent:
    """Agent responsible for managing model fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer agent with configuration."""
        self.config = config
        self.state = TrainerAgentState(
            config=config,
            training_config=config.get("training", {}),
            model_config=config.get("model", {})
        )
        self.trainer = None
        
    async def arun(self, state: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Run the training process asynchronously."""
        try:
            logger.info(format_agent_message("trainer", "Starting training process"))
            
            # Get training data from state
            train_data = state.get("artifacts", {}).get("train_data") if state else []
            
            # Initialize trainer
            self._initialize_trainer()
            
            # Start training
            self._train()
            
            # Save model
            self._save_model()
            
            # Get progress info
            progress = self.get_progress()
            
            # Update and yield state
            yield {
                "config": self.config,
                "messages": [f"Training complete. Model saved to {self.state.model_path}"],
                "current_stage": "training",
                "artifacts": {
                    "train_data": train_data,
                    "trained_model": self.state.model_path,
                    "training_stats": self.state.training_stats,
                    "checkpoints": self.state.checkpoint_paths
                },
                "metrics": self.state.metrics,
                "completed": False
            }
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(format_agent_message("trainer", error_msg))
            yield {
                "config": self.config,
                "messages": [error_msg],
                "current_stage": "error",
                "artifacts": {},
                "metrics": {},
                "error": error_msg,
                "completed": False
            }
    
    def _initialize_trainer(self) -> None:
        """Initialize trainer with comprehensive configuration support."""
        logger.info(format_agent_message("trainer", "Initializing trainer"))
        
        # Validate model configuration
        if not self.state.model_config.get("base_model_name"):
            raise AgentException("No base model specified in configuration")
            
        # Setup output directory with versioning
        output_dir = Path(self.state.training_config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create comprehensive training configuration
            training_config = TrainingConfig(
                # Model configuration
                model_name=self.state.model_config["base_model_name"],
                output_dir=str(output_dir),
                
                # Dataset configuration
                train_dataset=self.state.config.get("train_dataset"),
                eval_dataset=self.state.config.get("eval_dataset"),
                
                # Full configurations
                model_config=self.state.model_config,
                training_config=self.state.training_config,
                
                # Feature flags
                use_peft=self.state.training_config.get("use_peft", True),
                use_unsloth=self.state.training_config.get("use_unsloth", True)
            )
            
            # Initialize trainer with full configuration
            self.trainer = LLMTrainer(training_config)
            self.state.model_path = str(output_dir)
            
            # Log initialization details
            logger.info(format_agent_message(
                "trainer",
                f"Trainer initialized with model: {self.state.model_config['base_model_name']}"
            ))
            
        except Exception as e:
            raise AgentException(f"Failed to initialize trainer: {str(e)}")
    
    def _train(self) -> None:
        """Run the training process with comprehensive monitoring."""
        if not self.trainer:
            raise AgentException("Trainer not initialized")
            
        logger.info(format_agent_message("trainer", "Starting training process"))
        
        try:
            # Log training configuration
            logger.info(format_agent_message(
                "trainer",
                f"Training configuration: epochs={self.state.training_config.get('num_epochs')}, "
                f"batch_size={self.state.training_config.get('batch_size')}, "
                f"learning_rate={self.state.training_config.get('learning_rate')}"
            ))
            
            # Run training
            result = self.trainer.train()
            
            # Process and store results
            self._process_training_result(result)
            
            logger.info(format_agent_message(
                "trainer",
                f"Training completed in {result.get('train_runtime', 0):.2f} seconds"
            ))
            
        except Exception as e:
            raise AgentException(f"Training failed: {str(e)}")
    
    def _process_training_result(self, result: Dict[str, Any]) -> None:
        """Process and store comprehensive training results."""
        # Extract and store detailed training statistics
        self.state.training_stats = {
            "train_runtime": result.get("train_runtime", 0),
            "train_samples_per_second": result.get("train_samples_per_second", 0),
            "train_steps_per_second": result.get("train_steps_per_second", 0),
            "total_steps": result.get("total_steps", 0),
            "epoch": result.get("epoch", 0),
            "global_step": result.get("global_step", 0)
        }
        
        # Store training metrics
        if "metrics" in result:
            self.state.metrics = result["metrics"]
            logger.info(format_agent_message(
                "trainer",
                f"Training metrics: {json.dumps(result['metrics'], indent=2)}"
            ))
            
        # Track best model checkpoint
        if "best_model_checkpoint" in result:
            self.state.best_model_path = result["best_model_checkpoint"]
            logger.info(format_agent_message(
                "trainer",
                f"Best model checkpoint: {self.state.best_model_path}"
            ))
            
        # Save checkpoints information
        if "checkpoints" in result:
            self.state.checkpoint_paths = result["checkpoints"]
            logger.info(format_agent_message(
                "trainer",
                f"Saved {len(self.state.checkpoint_paths)} checkpoints"
            ))
    
    def _save_model(self) -> None:
        """Save the final model and comprehensive training artifacts."""
        if not self.trainer:
            return
            
        logger.info(format_agent_message("trainer", "Saving model and artifacts"))
        
        try:
            # Determine save format based on configuration
            save_format = "gguf" if self.state.training_config.get("use_unsloth", True) else "standard"
            
            # Save model
            self.trainer.save(self.state.model_path, save_format=save_format)
            logger.info(format_agent_message(
                "trainer",
                f"Model saved in {save_format} format"
            ))
            
            # Save comprehensive configuration
            config_path = os.path.join(self.state.model_path, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "model_config": self.state.model_config,
                    "training_config": self.state.training_config,
                    "data_config": self.state.config.get("data", {})
                }, f, indent=2)
            
            # Save detailed metrics
            metrics_path = os.path.join(self.state.model_path, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump({
                    "training_stats": self.state.training_stats,
                    "metrics": self.state.metrics,
                    "checkpoints": self.state.checkpoint_paths
                }, f, indent=2)
            
            logger.info(format_agent_message(
                "trainer",
                "Saved model configuration and metrics"
            ))
            
        except Exception as e:
            logger.error(format_agent_message(
                "trainer",
                f"Error saving model artifacts: {str(e)}"
            ))
            
    def get_progress(self) -> Dict[str, Any]:
        """Get comprehensive training progress information."""
        return {
            "current_epoch": self.state.current_epoch,
            "current_step": self.state.current_step,
            "metrics": self.state.metrics,
            "training_stats": self.state.training_stats,
            "checkpoint_paths": self.state.checkpoint_paths,
            "best_model_path": self.state.best_model_path,
            "output_dir": self.state.model_path,
            "model_name": self.state.model_config.get("base_model_name"),
            "training_config": {
                "batch_size": self.state.training_config.get("batch_size"),
                "learning_rate": self.state.training_config.get("learning_rate"),
                "num_epochs": self.state.training_config.get("num_epochs")
            }
        }
