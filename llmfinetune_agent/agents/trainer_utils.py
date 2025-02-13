"""Core training functionality for LLM fine-tuning."""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

logger = logging.getLogger(__name__)

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
    use_unsloth: bool = True

def create_training_arguments(
    output_dir: str,
    training_config: Dict[str, Any],
    use_unsloth: bool = True
) -> TrainingArguments:
    """Create training arguments based on configuration."""
    common_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": training_config.get("batch_size", 2),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 4),
        "warmup_steps": training_config.get("warmup_steps", 5),
        "num_train_epochs": training_config.get("num_epochs", 1),
        "learning_rate": training_config.get("learning_rate", 2e-4),
        "logging_steps": training_config.get("logging_steps", 1),
        "optim": "adamw_8bit",
        "weight_decay": training_config.get("weight_decay", 0.01),
        "lr_scheduler_type": training_config.get("lr_scheduler", "linear"),
        "seed": training_config.get("seed", 3407),
        "report_to": training_config.get("report_to", "none"),
    }

    if use_unsloth:
        common_args.update({
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
        })
    else:
        common_args.update({
            "fp16": training_config.get("fp16", False),
            "bf16": training_config.get("bf16", False),
            "max_grad_norm": training_config.get("max_grad_norm", 0.3),
            "warmup_ratio": training_config.get("warmup_ratio", 0.03),
            "evaluation_strategy": "steps" if training_config.get("eval_dataset") else "no",
            "save_strategy": "steps",
            "save_steps": training_config.get("save_steps", 100),
            "logging_steps": training_config.get("logging_steps", 10),
        })

    return TrainingArguments(**common_args)

def setup_unsloth_model(
    model_name: str,
    training_config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Setup model and tokenizer using unsloth."""
    max_seq_length = training_config.get("max_seq_length", 1024)
    load_in_4bit = training_config.get("load_in_4bit", True)
    dtype = training_config.get("dtype", None)  # Auto

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Setup PEFT configuration
    peft_config = training_config.get("peft", {})
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=peft_config.get("r", 16),
        target_modules=peft_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=peft_config.get("lora_alpha", 16),
        lora_dropout=peft_config.get("lora_dropout", 0),
        bias=peft_config.get("bias", "none"),
        use_gradient_checkpointing=peft_config.get("use_gradient_checkpointing", "unsloth"),
        random_state=peft_config.get("random_state", 3407),
        use_rslora=peft_config.get("use_rslora", False),
        loftq_config=peft_config.get("loftq_config", None),
    )

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=training_config.get("chat_template", "llama-3.1"),
    )

    return peft_model, tokenizer

def setup_standard_model(
    model_name: str,
    training_config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Setup model and tokenizer using standard HuggingFace approach."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup model configuration
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if training_config.get("load_in_4bit", False):
        compute_dtype = (torch.float16 if training_config.get("torch_dtype") == "float16"
                        else torch.bfloat16)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    if training_config.get("use_peft", True):
        # Prepare model for k-bit training if using quantization
        if training_config.get("load_in_4bit", False):
            model = prepare_model_for_kbit_training(model)

        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=training_config.get("lora_r", 8),
            lora_alpha=training_config.get("lora_alpha", 32),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=training_config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer

class LLMTrainer:
    """Trainer class for LLM fine-tuning with support for unsloth and standard approaches."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup model and tokenizer
        if config.use_unsloth:
            self.model, self.tokenizer = setup_unsloth_model(
                config.model_name,
                config.training_config or {}
            )
        else:
            self.model, self.tokenizer = setup_standard_model(
                config.model_name,
                config.training_config or {}
            )
            
    def train(self) -> Dict[str, Any]:
        """Run the training process."""
        training_args = create_training_arguments(
            self.config.output_dir,
            self.config.training_config or {},
            self.config.use_unsloth
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.config.train_dataset,
            eval_dataset=self.config.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.training_config.get("max_seq_length", 1024),
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        # Apply response-only training if configured
        if self.config.training_config.get("train_on_responses_only", True):
            self.trainer = train_on_responses_only(
                self.trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
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

    def save(self, output_dir: Optional[str] = None, save_format: str = "gguf"):
        """Save the model and tokenizer."""
        save_dir = output_dir or self.config.output_dir
        
        if save_format == "gguf" and hasattr(self.model, "save_pretrained_gguf"):
            self.model.save_pretrained_gguf(save_dir, self.tokenizer)
        else:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
