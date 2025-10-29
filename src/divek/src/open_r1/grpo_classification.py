# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from datasets import DatasetDict

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer #, Gemma3GRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
from rewards import format_reward, mcq_reward, accuracy_reward, format_reward_v2
from dataset import prepare_datasets

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'mcq'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    hard_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the hard dataset."},
    )
    use_hard_examples: bool = field(
        default=False,
        metadata={"help": "Whether to include hard examples in the training dataset."},
    )
    normal_to_hard_ratio: float = field(
        default=2,
        metadata={"help": "The ratio of normal to hard examples in the combined dataset."},
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the vision tower during training."},
    )
    freeze_text: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the text tower during training."},
    )

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "mcq": mcq_reward,
    "format_v2": format_reward_v2
}


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # import pdb; pdb.set_trace()

    # Prepare the dataset
    train_dataset = prepare_datasets(
        script_args,
        use_hard_examples=script_args.use_hard_examples,
        normal_to_hard_ratio=script_args.normal_to_hard_ratio
    )

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer

    # if "gemma-3" in model_args.model_name_or_path.lower():
    #     trainer_cls = Gemma3GRPOTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=DatasetDict.load_from_disk(script_args.dataset_name)[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        freeze_vision_tower=script_args.freeze_vision_tower,
        freeze_text=script_args.freeze_text,
    )

    # Train and push the model to the Hub
    if training_args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        print("Starting training from scratch.")
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
