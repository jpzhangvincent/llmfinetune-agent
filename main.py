"""Main entry point for the LLM fine-tuning framework."""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llmfinetune_agent import (
    OrchestratorAgent,
    DataGeneratorAgent,
    KnowledgeRetrievalAgent,
    TrainerAgent,
    EvaluatorAgent,
    DebuggerAgent,
    load_config
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Framework")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name from config"
    )
    
    parser.add_argument(
        "--train-file",
        type=str,
        help="Override training data file from config"
    )
    
    parser.add_argument(
        "--eval-file",
        type=str,
        help="Override evaluation data file from config"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--use-rlhf",
        action="store_true",
        help="Enable RLHF training"
    )
    
    return parser.parse_args()

def update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    if args.model:
        config["model"]["base_model_name"] = args.model
        
    if args.train_file:
        config["data"]["train_file"] = args.train_file
        
    if args.eval_file:
        config["data"]["validation_file"] = args.eval_file
        
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
        
    if args.use_rlhf:
        config["training"]["use_rlhf"] = True
        
    return config

def initialize_agents(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all agents."""
    agents = {
        "orchestrator": OrchestratorAgent(config),
        "data_generator": DataGeneratorAgent(config),
        "knowledge_retrieval": KnowledgeRetrievalAgent(config),
        "trainer": TrainerAgent(config),
        "evaluator": EvaluatorAgent(config),
        "debugger": DebuggerAgent(config)
    }
    return agents

def run_workflow(config: Dict[str, Any], agents: Dict[str, Any]) -> None:
    """Run the fine-tuning workflow."""
    try:
        # Initialize orchestrator
        orchestrator = agents["orchestrator"]
        
        # Initialize workflow state with data
        train_file = config["data"]["train_file"]
        val_file = config["data"]["validation_file"]
        
        logger.info(f"Loading data from {train_file} and {val_file}")
        
        # Start with data preparation
        data_agent = agents["data_generator"]
        data_state = data_agent.run()
        
        if data_state.error:
            raise Exception(f"Data preparation failed: {data_state.error}")
            
        # Update training config with prepared data
        trainer_agent = agents["trainer"]
        trainer_agent.state.config.update({
            "train_dataset": data_state.dataset["train"],
            "eval_dataset": data_state.dataset.get("validation")
        })
        
        # Run training
        logger.info("Starting training process")
        trainer_state = trainer_agent.run()
        
        if trainer_state.error:
            raise Exception(f"Training failed: {trainer_state.error}")
            
        # Run evaluation
        evaluator_agent = agents["evaluator"]
        evaluator_agent.state.model_path = trainer_state.model_path
        evaluator_agent.state.config.update({
            "eval_dataset": data_state.dataset.get("validation")
        })
        
        logger.info("Starting evaluation")
        eval_state = evaluator_agent.run()
        
        if eval_state.error:
            raise Exception(f"Evaluation failed: {eval_state.error}")
            
        # Run debugging if needed
        if eval_state.metrics_results:
            debugger_agent = agents["debugger"]
            debugger_agent.state.evaluation_results = {
                "metrics": eval_state.metrics_results,
                "error_analysis": eval_state.error_analysis
            }
            
            logger.info("Running performance analysis")
            debug_state = debugger_agent.run()
            
            if debug_state.suggestions:
                logger.info("Improvement suggestions:")
                for suggestion in debug_state.suggestions:
                    logger.info(f"- {suggestion['suggestion']} (Priority: {suggestion['priority']})")
                    
        logger.info("Workflow completed successfully")
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load and update configuration
        config = load_config(args.config)
        config = update_config(config, args)
        
        # Initialize agents
        agents = initialize_agents(config)
        
        # Run workflow
        run_workflow(config, agents)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
