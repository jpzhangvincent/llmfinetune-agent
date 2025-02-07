# LLM Fine-tuning Agent Framework

A multi-agent framework for automated LLM fine-tuning using langgraph, supporting both supervised fine-tuning and RLHF.

## Features

- Multi-agent architecture for coordinated fine-tuning
- Support for supervised fine-tuning and RLHF (Reinforcement Learning from Human Feedback)
- Automated data generation and preparation
- Optional knowledge retrieval for enhanced training
- Comprehensive evaluation and debugging
- Easy configuration through JSON files
- Built with langgraph for robust workflow management

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llmfinetune-agent.git
cd llmfinetune-agent

# Create a new virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using uv
pip install uv
uv pip install -r requirements.txt
```

## Quick Start

1. Configure your model and training settings in `configs/default_config.json`
2. Prepare your training data in JSON format similar to:
```json
[
    {
        "input": "Your input text",
        "output": "Expected output text"
    }
]
```
3. Run the fine-tuning:
```bash
python main.py --config configs/default_config.json
```

## Configuration

The framework uses a JSON configuration file with the following main sections:

```json
{
    "model": {
        "base_model_name": "meta-llama/Llama-2-7b-hf",
        "model_type": "causal_lm",
        "max_length": 2048,
        "torch_dtype": "float16",
        "load_in_4bit": true
    },
    "training": {
        "output_dir": "outputs",
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "use_rlhf": false
    },
    "data": {
        "train_file": "data/train.json",
        "validation_file": "data/validation.json"
    }
}
```

## Command Line Arguments

- `--config`: Path to configuration file (default: configs/default_config.json)
- `--model`: Override model name from config
- `--train-file`: Override training data file from config
- `--eval-file`: Override evaluation data file from config
- `--output-dir`: Override output directory from config
- `--use-rlhf`: Enable RLHF training

## Architecture

The framework consists of several specialized agents:

1. **Orchestrator Agent**: Coordinates the entire fine-tuning workflow
2. **Data Generator Agent**: Handles data preparation and synthetic data generation
3. **Knowledge Retrieval Agent**: Retrieves relevant knowledge for training (optional)
4. **Trainer Agent**: Manages the fine-tuning process
5. **Evaluator Agent**: Assesses model performance
6. **Debugger Agent**: Analyzes issues and suggests improvements

These agents work together in a coordinated workflow managed by langgraph.

## RLHF Support

The framework supports RLHF training through the following components:

- Preference data handling
- PPO training integration
- Reward modeling
- Human feedback integration

Enable RLHF by setting `use_rlhf: true` in the config or using the `--use-rlhf` flag.

## Evaluation

The framework supports multiple evaluation metrics:

- ROUGE scores
- BLEU scores
- BERT Score
- Custom metrics can be added through the evaluation configuration

## Development

Run tests:
```bash
pytest tests/ -v
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- langgraph
- trl (for RLHF)
- accelerate
- Other dependencies listed in requirements.txt

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
