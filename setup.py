"""Setup script for llmfinetune-agent package."""
from setuptools import find_packages, setup

setup(
    name="llmfinetune-agent",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "wandb>=0.16.0",
        "peft>=0.7.0",
        "trl>=0.7.4",
        "bitsandbytes==0.42",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "autotrain-advanced>=0.0.1",
        "langgraph>=0.2.70",
        "evaluate>=0.4.3",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.1.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
            "ruff>=0.1.15",
        ]
    }
)
