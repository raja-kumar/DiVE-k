import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, LlavaForConditionalGeneration, LlavaNextProcessor
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
# from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration#, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from prompts import PROMPTS

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool
import argparse

import random
random.seed(21)
# from utils import get_cat_name_from_json

def plot_images(image_paths):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Top-K Accuracy Evaluation")
    parser.add_argument("--model_root", type=str, default="/app/saved_models/vrft/CUB_200_2011/")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--exp_name", type=str, default="Qwen2_5-VL-7B-Instruct_GRPO_cub_base_and_hard_mcq")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-400")
    parser.add_argument("--num_shot", type=int, default=0)
    parser.add_argument("--eval_type", type=str, default="baseline")
    parser.add_argument("--data_root", type=str, default="/data2/raja/")
    parser.add_argument("--dataset", type=str, default="CUB_200_2011")
    parser.add_argument("--split", type=str, default="new_val")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_cat_list", type=str, default="True", help="Whether to use category list in the prompt")
    return parser.parse_args()

args = parse_args()

MODEL_ROOT = args.model_root
BASE_MODEL = args.base_model
EXP_NAME = args.exp_name
CHECKPOINT = args.checkpoint
# zero_shot = args.zero_shot.lower() == "true"
num_shot = args.num_shot
eval_type = args.eval_type
DATA_ROOT = args.data_root
dataset = args.dataset
split = args.split
max_new_tokens = args.max_new_tokens
use_cat_list = args.use_cat_list.lower() == "true"


model_path = os.path.join(MODEL_ROOT, f"{EXP_NAME}", CHECKPOINT)  # full path to the model"
print(GREEN + "Model path: " + model_path + RESET)
model_base = BASE_MODEL  # base model name

if EXP_NAME == "baseline":
    model_path = BASE_MODEL

split_name = split.split("_")[0]

if (num_shot>0):
    data_json_path = f"{DATA_ROOT}/{dataset}/fewshot/{num_shot}_shots_all_val.json"
    category_file = f"{DATA_ROOT}/{dataset}/fewshot/all_categories.txt"
else:
    data_json_path = f"{DATA_ROOT}/{dataset}/zero_shot/subsample_{split}.json"
    category_file = f"{DATA_ROOT}/{dataset}/zero_shot/{split_name}_categories.txt"

# zero_shot_json_path = f"{DATA_ROOT}/{dataset}/zero_shot/subsample_{split}.json"
logger.info(f"using category file {category_file}")
output_path = f"./output/{dataset}/{eval_type}/"

if "checkpoint" in model_path:
    model_name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1] # use checkpoint name
else:
    model_name = model_path.split("/")[-1]  # model name

data_name = data_json_path.split("/")[-1].split(".")[0]  # data name
output_file = f"{model_name}_{data_name}_{use_cat_list}.json"  # output file name

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file_path = os.path.join(output_path, output_file)

print(GREEN + "output path" + output_file_path + RESET)
output_data = {}

# split_name = split.split("_")[0] 
# category_file = f"{DATA_ROOT}/{dataset}/zero_shot/{split_name}_categories.txt"

with open(category_file, 'r') as f:
    categories = f.read().splitlines()

def run(rank, world_size):

    local_output_data = {}

    if "Qwen2.5" in model_base:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_base) 
    elif "llava" in model_base:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = LlavaNextProcessor.from_pretrained(model_base)
    elif "Phi-3.5" in model_base:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
        )
        processor = AutoProcessor.from_pretrained(model_base, 
            trust_remote_code=True, 
            num_crops=16
        )

        print(GREEN + "Using Phi-3.5 model" + RESET)
    elif "gemma-3" in model_base:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_base)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        

    model = model.to(torch.device(rank))
    model = model.eval()

    with open(data_json_path, 'r') as f:
        infer_data = json.load(f)
    
    random.seed(21)
    random.shuffle(infer_data)

    # infer_data = infer_data[:10]

    print(GREEN + "Number of images in infer data: " + str(len(infer_data)) + RESET)
    

    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(infer_data)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = infer_data[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))
    

    error_count = 0
    right_count = 0
    for item in tqdm(split_images, total=len(split_images), desc=f"Rank {rank} Processing"):
        image_path = item['image_path']
        image_label = item['solution']

        prompt = item['problem']
        image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/fgvc_aircraft/", 
                        DATA_ROOT)
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", 
                        DATA_ROOT)

        temp, answer_format, data_type = PROMPTS[dataset]["instruction"], PROMPTS[dataset]["answer_format"], PROMPTS[dataset]["data_name"]

        if use_cat_list:
            # question = (
            # f"This is an image containing a {data_type}. {temp}\n"
            # f"the {answer_format} of the {data_type} strictly belongs to below category list {categories}.\n"
            # "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            # "The output answer format should be as follows:\n"
            # f"<think> ... </think> <answer>{answer_format}</answer>\n"
            # "Please strictly follow the format."
            # )

            question = (
            f"This is an image containing a {data_type}. {temp}\n"
            f"the {answer_format} of the {data_type} strictly belongs to below category list {categories}.\n"
            "You should first think step by step to analyze the image and then give your final answer."
            "During the thinking process first shortlist top5 possible categories from the category list that you think are most relevant to the image."
            "Then among those shortlisted categories, perform a differentiation analysis to find the most suitable category that matches the image content."
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            f"<think> ... </think> <answer>{answer_format}</answer>\n"
            "Please strictly follow the format."
            )
        else:
            question = prompt
        
        if eval_type == "sft":
            question = (
                f"This is an image containing a {data_type}. {temp}\n"
                f"the {answer_format} of the {data_type} strictly belongs to below category list {categories}.\n"
                f"Only output the name of the species without any additional text."
            )
        print(RED + question + RESET)
    
        query = "<image>\n"+question
        # print(RED+query+RESET)

        if "llava" in model_base:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            raw_image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=text,
                images=raw_image,
                return_tensors="pt",
                image_sizes=raw_image.size,
            )
        elif "Phi-3.5" in model_base:
            images = [Image.open(image_path)]
            query = "<|image_1|>\n" + question

            messages = [
                {"role": "user", "content": query},
            ]

            prompt = processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = processor(prompt, images, return_tensors="pt")
        elif "gemma-3" in model_base:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
            )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": query}
                    ]
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, 
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output

        # print(GREEN + "before generate" + RESET)
        # generation_args = { 
        #     "max_new_tokens": 1000, 
        #     "temperature": 0.0, 
        #     "do_sample": False, 
        # } 

        # generated_ids = model.generate(**inputs, 
        # eos_token_id=processor.tokenizer.eos_token_id, 
        # **generation_args
        # )
        generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)

        # print(GREEN + "Generated IDs: " + str(generated_ids) + RESET)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        # print("\033[92m" + response + "\033[0m")
        image_id = image_path.split("/")[-1].split(".")[0]

        try:
            if eval_type == "sft":
                # For SFT, search in complete response without parsing

                local_output_data[image_id] = {
                    "groundtruth": image_label,
                    "reasoning": "", # No reasoning for SFT
                    "answer": response
                }

                image_label = image_label.replace(' ','').replace('_','').lower()
                response_lower = response.replace(' ','').replace('_','').lower()

                if image_label in response_lower:
                    right_count += 1
                else:
                    error_count += 1
            else:
                # For other cases, keep the original parsing logic
                reasoning = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                reasoning_content = reasoning.group(1).strip() if reasoning else ""
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)\n</answer>", response, re.DOTALL)
                
                answer_content = match.group(1)

                local_output_data[image_id] = {
                    "groundtruth": image_label,
                    "reasoning": reasoning_content,
                    "answer": answer_content
                }

                # print(local_output_data[image_id])
                if ("describe" in model_path):
                    # For describe task, we use the image_id as the key
                    describe_match = re.search(r'<describe>(.*?)</describe>', response, re.DOTALL)
                    if describe_match:
                        describe_content = describe_match.group(1).strip()
                    else:
                        describe_content = ""
                    
                    rethink_match = re.search(r'<rethink>(.*?)</rethink>', response, re.DOTALL)
                    if rethink_match:
                        rethink_content = rethink_match.group(1).strip()
                    else:
                        rethink_content = ""
                    
                    local_output_data[image_id]["describe"] = describe_content
                    local_output_data[image_id]["rethink"] = rethink_content
        except Exception as e:
            print(RED + "Error in processing response: " + response + RESET)
            local_output_data[image_id] = {
                    "groundtruth": image_label,
                    "reasoning": "",
                    "answer": ""
                }
            error_count += 1
        
    return [error_count, right_count, local_output_data]

def main():
    multiprocess = torch.cuda.device_count() >= 1
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_count_right = 0
        global_results = []
        for i in range(world_size):
            logger.info('Rank: ' + str(i) + ' Error Number: ' + str(result_lists[i][0]) + 
                        ' Right Number: ' + str(result_lists[i][1]))
            global_count_error += int(result_lists[i][0])
            global_count_right = global_count_right + result_lists[i][1]

            output_data.update(result_lists[i][2])  # merge local output data
            
        logger.info('Error number: ' + str(global_count_error))  
        logger.info('Total Right Number: ' + str(global_count_right))
        logger.info("above count holds meaning only for sft eval. IGNORE for other evals.")
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()

    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Output saved to {output_file_path}")