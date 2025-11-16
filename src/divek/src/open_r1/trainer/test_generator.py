from generator import TopKGenerator
from datasets import interleave_datasets, concatenate_datasets
from datasets import DatasetDict
import torch
import random
import json
import re

from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2VLForConditionalGeneration
    # Gemma3ForConditionalGeneration, Gemma3Processor  # Uncomment if using
)


dataset_name = "/data2/datasets/CUB_200_2011/zero_shot/subsample_base_train_dataset/"
dataset = DatasetDict.load_from_disk(dataset_name)

# Format normal dataset
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ],
    }

print(dataset.keys())
if "image" in dataset['train'].features:
    print("DEBUG: Normal dataset contains images.")
    dataset = dataset.map(make_conversation_image)
else:
    print("DEBUG: Normal dataset does not contain images.")
    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns("messages")


model_base = "Qwen/Qwen2.5-VL-7B-Instruct"
model_path = model_base
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )

model = model.to("cuda:0")
processor = AutoProcessor.from_pretrained(model_base) 
# processor = processor.to("cuda:0")

class_to_idx_path = "/data2/datasets/CUB_200_2011/class_2_idx.json"
base_categories_path = "/data2/datasets/CUB_200_2011/zero_shot/base_categories.txt"
with open(class_to_idx_path, 'r') as f:
    class_to_idx = json.load(f)
data_name = "CUB_200_2011"

temp = TopKGenerator(class_to_idx=class_to_idx, data_name=data_name, processing_class=processor, category_names_path=base_categories_path)

train = dataset["train"]

for batch in train:
    # batch is a dict of lists
    #print(batch.keys())
    print(train[0])
    prompt, answer = temp.get_topk(model, [train[0]])
    break


# print("Predictions:", curr_pred)
print("Prompt:", prompt)
print("Answer:", answer)