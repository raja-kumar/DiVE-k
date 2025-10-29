import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

# --- Argument Parsing ---
# We use HfArgumentParser to parse arguments into dataclasses.
# This is the standard way to handle arguments in Hugging Face's examples.

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation to use (e.g., 'flash_attention_2' or 'sdpa')."},
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to use.
    """
    dataset_name: str = field(
        default="dummy",
        metadata={"help": "The name of the dataset to use (e.g., a path to a file or 'dummy')."}
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "The maximum length of the prompt."}
    )

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Setup ---
    # The 'local_rank' is automatically passed by torchrun.
    # It's important for DeepSpeed to know which process is which.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == 0:
        print("All parsed arguments:")
        print(model_args)
        print(data_args)
        print(training_args)


    # --- Load Model and Tokenizer ---
    print(f"[Rank {local_rank}] Loading model: {model_args.model_name_or_path}")

    # Determine torch dtype based on training arguments
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)

    # When using DeepSpeed ZeRO-3, we initialize the model under a special context manager.
    # This tells DeepSpeed to shard the model as it's being created, saving massive amounts of memory.
    # If not using deepspeed, the model is loaded as usual.
    model_load_kwargs = {}
    if training_args.deepspeed:
        # With ZeRO-3, there's no need to specify device_map, as DeepSpeed handles device placement.
        pass
    else:
        # For other setups, you might use device_map
        model_load_kwargs['device_map'] = {'': local_rank}


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        **model_load_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"[Rank {local_rank}] Model and Tokenizer loaded successfully.")

    # --- Load and Preprocess Dataset ---
    # For this example, we create a dummy dataset.
    # In a real use case, you would load from `data_args.dataset_name`.
    print(f"[Rank {local_rank}] Creating dummy dataset.")
    dummy_data = {
        "prompt": [
            "The quick brown fox jumps over the lazy dog.",
            "DeepSpeed is a deep learning optimization library.",
            "Hugging Face Transformers provides thousands of pretrained models.",
            "Fine-tuning a model on a specific task can greatly improve its performance.",
            "torchrun is the recommended tool for launching PyTorch distributed jobs.",
        ] * 100 # Make the dataset large enough for training
    }
    dataset = Dataset.from_dict(dummy_data)

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=data_args.max_prompt_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    # The Trainer for Causal LM expects 'labels'
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])


    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # --- Train ---
    print(f"[Rank {local_rank}] Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print(f"[Rank {local_rank}] Training finished successfully.")


    # --- Save Final Model ---
    # The `save_model` call is smart enough to handle DeepSpeed ZeRO-3.
    # It will gather the sharded model weights onto the CPU of the rank 0 process
    # before saving the consolidated state_dict.
    print(f"[Rank {local_rank}] Saving final model...")
    trainer.save_model()
    print(f"[Rank {local_rank}] Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()