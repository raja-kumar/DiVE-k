# train_with_deepspeed.py
#
# This script provides a minimal working example of how to train a Hugging Face
# model using DeepSpeed. It includes functionality for saving checkpoints and
# reloading the model state to resume training.
#
# To run this script, you need to have PyTorch, Transformers, and DeepSpeed installed.
#
# Installation:
# pip install torch transformers deepspeed
#
# How to run:
# deepspeed train_with_deepspeed.py --deepspeed_config ds_config.json
#
# To resume from a checkpoint:
# deepspeed train_with_deepspeed.py --deepspeed_config ds_config.json --load_from_checkpoint ./checkpoints

import torch
import deepspeed
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def create_dummy_dataset(num_samples=100, seq_length=128, vocab_size=50257):
    """Creates a dummy dataset for demonstration purposes."""
    features = torch.randint(low=0, high=vocab_size, size=(num_samples, seq_length))
    labels = torch.randint(low=0, high=vocab_size, size=(num_samples, seq_length))
    return torch.utils.data.TensorDataset(features, labels)

def main():
    """Main training routine."""
    # --- Argument Parsing ---
    # DeepSpeed automatically handles its own arguments, but we can add our own.
    parser = argparse.ArgumentParser(description="DeepSpeed Hugging Face Training Example")
    parser.add_argument('--load_from_checkpoint',
                        type=str,
                        default=None,
                        help='Directory to load a model checkpoint from.')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='./checkpoints',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=2,
                        help='Number of training epochs.')
    parser.add_argument('--model_name',
                        type=str,
                        default='gpt2',
                        help='Name of the Hugging Face model to use.')

    # Add DeepSpeed-specific arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # --- Model and Tokenizer Initialization ---
    print("Initializing model and tokenizer...")
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset Creation ---
    print("Creating a dummy dataset...")
    # In a real scenario, you would load your own dataset here.
    train_dataset = create_dummy_dataset(vocab_size=tokenizer.vocab_size)

    # --- DeepSpeed Initialization ---
    print("Initializing DeepSpeed...")
    # The `deepspeed.initialize` function wraps the model, optimizer, and dataloader.
    # It requires the model, command-line arguments, and a dataloader.
    # The optimizer is created internally by DeepSpeed based on the config file.
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset
    )

    # --- Checkpoint Loading ---
    # If a checkpoint directory is provided, load the state.
    if args.load_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.load_from_checkpoint}")
        # The load_checkpoint method returns the checkpoint tag, which can be useful
        # for tracking progress (e.g., which epoch/step it was saved at).
        load_path, client_sd = model_engine.load_checkpoint(args.load_from_checkpoint)
        if load_path is None:
            print(f"Warning: Could not find checkpoint at {args.load_from_checkpoint}")
        else:
            print(f"Successfully loaded checkpoint from {load_path}")


    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_loader):
            # Move batch to the correct device
            batch = [t.to(model_engine.device) for t in batch]
            inputs, labels = batch

            # Forward pass
            outputs = model_engine(inputs, labels=labels)
            loss = outputs.loss

            # Backward pass - DeepSpeed handles the backward pass and gradient accumulation
            model_engine.backward(loss)

            # Optimizer step - DeepSpeed handles the optimizer step
            model_engine.step()

            # Print progress
            if model_engine.global_rank == 0: # Only print on the main process
                print(f"Epoch: {epoch+1}/{args.num_epochs}, Step: {step+1}, Loss: {loss.item():.4f}")

    # --- Checkpoint Saving ---
    # After training, save the final model state.
    print(f"Saving final model checkpoint to {args.checkpoint_dir}")
    # The save_checkpoint method takes the save directory and a unique tag (e.g., epoch/step).
    # This tag is used to create a subdirectory for the checkpoint.
    model_engine.save_checkpoint(args.checkpoint_dir, tag="final_model")

    print("Training finished.")

if __name__ == "__main__":
    main()