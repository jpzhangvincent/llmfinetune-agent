"""Main entry point for the LLM fine-tuning framework."""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, AsyncIterator

from dotenv import load_dotenv
from langgraph.graph import StateGraph

from llmfinetune_agent import (
    DataGenerator,
    DebuggerAgent,
    DebuggerState,
    Evaluator,
    KnowledgeRetrievalAgent,
    KnowledgeRetrievalState,
    Orchestrator,
    TrainerAgent,
    TrainerAgentState,
    load_config,
)
from llmfinetune_agent.agents.utils import AgentState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentWorkflow:
    """Manages the fine-tuning workflow using langraph."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = None
        self.agents = {}
        self._initialize_agents()
        self._build_graph()
    
    def _initialize_agents(self) -> None:
        """Initialize all agents."""
        self.agents = {
            "orchestrator": Orchestrator(self.config),
            "data_generator": DataGenerator(self.config),
            "knowledge_retrieval": KnowledgeRetrievalAgent(self.config),
            "trainer": TrainerAgent(self.config),
            "evaluator": Evaluator(self.config),
            "debugger": DebuggerAgent(self.config)
        }
    
    def _build_graph(self) -> None:
        """Build the langraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("orchestrator", self._create_node("orchestrator", AgentState))
        workflow.add_node("data_generator", self._create_node("data_generator", AgentState))
        workflow.add_node("knowledge_retrieval", self._create_node("knowledge_retrieval", AgentState))
        workflow.add_node("trainer", self._create_node("trainer", AgentState))
        workflow.add_node("evaluator", self._create_node("evaluator", AgentState))
        workflow.add_node("debugger", self._create_node("debugger", AgentState))
       
        # Add end node
        async def end_node(state: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
            # Get or create orchestrator state
            orchestrator_state = state.get("orchestrator")
            if orchestrator_state is None:
                orchestrator_state = AgentState(
                    config=self.config,
                    messages=["Workflow completed"],
                    current_stage="completed",
                    completed=True,
                    artifacts={},
                    metrics={}
                )
            elif isinstance(orchestrator_state, dict):
                orchestrator_state = AgentState(**orchestrator_state)
            
            # Update state
            orchestrator_state.completed = True
            orchestrator_state.current_stage = "completed"
            orchestrator_state.messages.append("Workflow completed")
            
            # Yield updated state
            yield {"orchestrator": orchestrator_state.__dict__}
        workflow.add_node("end", end_node)

        # Define edges
        workflow.set_entry_point("orchestrator")

        async def determine_next_node(state: Dict[str, Any]):
            # Get or create orchestrator state
            orchestrator_state = state.get("orchestrator")
            if orchestrator_state is None:
                orchestrator_state = AgentState(
                    config=self.config,
                    messages=[],
                    current_stage="start",
                    artifacts={},
                    metrics={}
                )
            elif isinstance(orchestrator_state, dict):
                orchestrator_state = AgentState(**orchestrator_state)
            
            # Get next stage
            next_stage = await self.agents["orchestrator"].determine_next_stage(orchestrator_state.__dict__)
            return next_stage

        workflow.add_conditional_edges(
            "orchestrator",
            determine_next_node,
            {
                "data_generator": "data_generator",
                "knowledge_retrieval": "knowledge_retrieval",
                "trainer": "trainer",
                "evaluator": "evaluator",
                "debugger": "debugger",
                "end": "end",
            }
        )

        workflow.add_edge("data_generator", "orchestrator")
        workflow.add_edge("knowledge_retrieval", "orchestrator")
        workflow.add_edge("trainer", "orchestrator")
        workflow.add_edge("evaluator", "orchestrator")
        workflow.add_edge("debugger", "orchestrator")

        self.graph = workflow.compile()

    def _create_node(self, agent_name: str, state_class):
        """Create a langraph node for an agent."""
        agent = self.agents[agent_name]

        async def node_func(state: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
            try:
                # Get or create agent state
                agent_state = state.get(agent_name)
                if agent_state is None:
                    agent_state = AgentState(
                        config=self.config,
                        messages=[],
                        current_stage=state.get("current_stage", "start"),
                        artifacts={},
                        metrics={}
                    )
                elif isinstance(agent_state, dict):
                    agent_state = AgentState(**agent_state)
                
                # Run agent
                async for result in agent.arun(agent_state.__dict__):
                    # Update state with result
                    if isinstance(result, dict):
                        for key in ["messages", "current_stage", "artifacts", "metrics", "completed", "error"]:
                            if key in result:
                                setattr(agent_state, key, result[key])
                    
                    # Always ensure config is present
                    agent_state.config = self.config
                    
                    # Yield updated state
                    yield {agent_name: agent_state.__dict__}

            except Exception as e:
                logger.error(f"Error in {agent_name}: {str(e)}")
                error_state = AgentState(
                    config=self.config,
                    messages=[f"Error in {agent_name}: {str(e)}"],
                    current_stage="error",
                    error=str(e)
                )
                yield {agent_name: error_state.__dict__}

        return node_func
    
    async def run(self, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the workflow."""
        if initial_state is None:
            initial_state = {"orchestrator": AgentState(config=self.config, messages=[], current_stage="start", artifacts={}, metrics={})}
        else:
            # Ensure initial state includes AgentState
            if "orchestrator" not in initial_state:
                initial_state["orchestrator"] = AgentState(config=self.config, messages=[], current_stage="start", artifacts={}, metrics={})
            else:
                # Initialize AgentState if a dict is passed
                if isinstance(initial_state["orchestrator"], dict):
                    initial_state["orchestrator"] = AgentState(**initial_state["orchestrator"])

        try:
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            return {"error": str(e)}

def parse_args() -> argparse.Namespace:
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
    
    parser.add_argument(
        "--test-agent",
        type=str,
        choices=["data_generator", "knowledge_retrieval", "trainer", "evaluator", "debugger"],
        help="Test a specific agent"
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

async def test_agent(agent_name: str, config: Dict[str, Any]) -> None:
    """Run tests for a specific agent."""
    try:
        # Initialize agent
        agent_class = {
            "data_generator": DataGenerator,
            "knowledge_retrieval": KnowledgeRetrievalAgent,
            "trainer": TrainerAgent,
            "evaluator": Evaluator,
            "debugger": DebuggerAgent
        }[agent_name]
        
        agent = agent_class(config)
        
        # Create test state
        test_state = AgentState(config=config, messages=[], current_stage="start", artifacts={}, metrics={})

        # Run agent
        agent = agent.compile()
        result = await agent.ainvoke({agent_name: test_state})
        logger.info(f"Test result for {agent_name}: {result}")

    except Exception as e:
        logger.error(f"Error testing {agent_name}: {str(e)}")
        raise

async def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    args = parse_args()
    
    try:
        # Load and update configuration
        config = load_config(args.config)
        config = update_config(config, args)
        
        if args.test_agent:
            await test_agent(args.test_agent, config)
            return
            
        # Initialize and run workflow
        workflow = AgentWorkflow(config)
        result = await workflow.run()
        
        if result.get("error"):
            logger.error(f"Workflow failed: {result['error']}")
        else:
            logger.info("Workflow completed successfully")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
