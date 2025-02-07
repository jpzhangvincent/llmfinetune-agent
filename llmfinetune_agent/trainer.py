"""Core training functionality for LLM fine-tuning."""
from typing import Any, Dict, Optional, Union
import os
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import (
    SFTTrainer,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead
)
from datasets import Dataset

@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str
    output_dir: str
    train_dataset: Union[Dataset, str]
    eval_dataset: Optional[Union[Dataset, str]] = None
    model_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    use_peft: bool = True
    use_rlhf: bool = False

class LLMTrainer:
    """Trainer class for LLM fine-tuning with support for supervised and RLHF."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Setup model configuration
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if config.model_config:
            if config.model_config.get("load_in_4bit", False):
                compute_dtype = (torch.float16 if config.model_config.get("torch_dtype") == "float16"
                               else torch.bfloat16)
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        if config.use_peft:
            # Prepare model for k-bit training if using quantization
            if config.model_config.get("load_in_4bit", False):
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA configuration
            lora_config = LoraConfig(
                r=config.training_config.get("lora_r", 8),
                lora_alpha=config.training_config.get("lora_alpha", 32),
                target_modules=["q_proj", "v_proj"],
                lora_dropout=config.training_config.get("lora_dropout", 0.1),
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            
    def train(self):
        """Run the training process."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.training_config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=self.config.training_config.get("gradient_accumulation_steps", 4),
            num_train_epochs=self.config.training_config.get("num_train_epochs", 3),
            learning_rate=self.config.training_config.get("learning_rate", 2e-5),
            max_grad_norm=self.config.training_config.get("max_grad_norm", 0.3),
            warmup_ratio=self.config.training_config.get("warmup_ratio", 0.03),
            evaluation_strategy="steps" if self.config.eval_dataset else "no",
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            remove_unused_columns=True
        )
        
        if self.config.use_rlhf:
            # Setup PPO training
            ppo_config = PPOConfig(
                learning_rate=1e-5,
                mini_batch_size=4,
                batch_size=16,
                gradient_accumulation_steps=4,
                optimize_cuda_cache=True,
            )
            
            # Add value head for RLHF
            model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model
            )
            
            self.trainer = PPOTrainer(
                config=ppo_config,
                model=model_with_value_head,
                tokenizer=self.tokenizer,
                dataset=self.config.train_dataset,
            )
        else:
            # Regular supervised fine-tuning
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.config.train_dataset,
                eval_dataset=self.config.eval_dataset,
                args=training_args,
                tokenizer=self.tokenizer,
            )
            
        # Run training
        train_result = self.trainer.train()
        self.trainer.save_model()
        
        # Save training state
        if self.config.output_dir:
            self.trainer.state.save_to_json(
                os.path.join(self.config.output_dir, "trainer_state.json")
            )
            
        return train_result
        
    def save(self, output_dir: Optional[str] = None):
        """Save the model and tokenizer."""
        save_dir = output_dir or self.config.output_dir
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
