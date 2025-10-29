import torch
import deepspeed
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# 1. --- Argument Parsing ---
# This is a standard practice for running Python scripts from the command line.
# DeepSpeed requires some of its own arguments, which we add here.
def get_args():
    parser = argparse.ArgumentParser(description="Simple DeepSpeed Training Example")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Add other arguments here as needed
    
    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    return args

# 2. --- Model and Tokenizer ---
# We'll use a relatively small pre-trained model from Hugging Face.
def get_model_and_tokenizer(model_name="gpt2"):
    """Loads the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set the padding token if it's not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

# 3. --- Dummy Dataset ---
# In a real-world scenario, you would load your data from a file or a Hugging Face dataset.
# For this prototype, we'll just use a small list of sentences.
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts
        # Tokenize the texts and store them
        self.encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return a dictionary of the tokenized inputs
        item = {key: val[idx] for key, val in self.encodings.items()}
        # The labels are the same as the input_ids for language modeling
        item['labels'] = item['input_ids'].clone()
        return item

# 4. --- Main Execution Block ---
if __name__ == "__main__":
    args = get_args()

    # --- Initialization ---
    model, tokenizer = get_model_and_tokenizer()
    
    # --- Create Dataset ---
    dummy_texts = [
        "Hello, this is a test sentence for the training script.",
        "DeepSpeed makes distributed training much easier.",
        "We are training a small GPT-2 model.",
        "This is the final sentence in our dummy dataset."
    ]
    train_dataset = SimpleDataset(tokenizer, dummy_texts)

    # --- DeepSpeed Configuration ---
    # This is where you define how DeepSpeed should optimize your training.
    # For this example, we'll use a very basic configuration.
    # In a real project, you'd typically load this from a JSON file.
    ds_config = {
        "train_batch_size": 4,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        }
    }

    # --- DeepSpeed Initialization ---
    # This is the core of DeepSpeed. It wraps your model, optimizer, etc.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(3):  # Train for 3 epochs
        for i, batch in enumerate(train_dataset):
            # Move batch to the correct device (GPU)
            # The local_rank is used to determine which GPU to use.
            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}
            
            # The batch from our simple dataset is a single item, so we need to unsqueeze it
            # to create a batch dimension.
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss

            # Backward pass
            model_engine.backward(loss)

            # Optimizer step
            model_engine.step()

            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")

    print("Training finished.")

    # To run this script, you would use the deepspeed command-line launcher:
    # deepspeed main.py