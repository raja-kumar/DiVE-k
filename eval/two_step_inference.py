"""
- load the trained model to get the pass@20 output
    - should use the category list
- clean this output to include 5 options only (with valid names)
    - only clean to exclude long options and species names
- run the trained model to get the MCQ output
- post process to get the category name and match
"""
import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
                        #   Gemma3ForConditionalGeneration, Gemma3Processor)
from transformers.generation import GenerationConfig
from peft import PeftModel
torch.manual_seed(1234)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import math
from prompts import PROMPTS
from utils import post_process_passk, extract_choice, clean_string

import random
random.seed(21)


RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class TwoStepInference:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def get_pass_at_20(self, inputs, generation_args):
        # Inference: Generation of the output

        try:
            generated_ids = self.model.generate(**inputs, **generation_args)
        except Exception as e:
            print(RED + "Error in generating output: " + RESET)
            print(RED + str(e) + RESET)
            return {}
        
        input_id_length = inputs.input_ids.shape[1] # Length of the single input sequence
        num_sequences = generation_args["num_return_sequences"]

        # curr_pred = set()  # Use a set to avoid duplicates
        curr_pred = {}

        for i in range(num_sequences):
            # Extract each generated sequence
            # Each generated sequence in `generated_ids` will start with the input_id_length
            # and then the new tokens.
            start_index = i * generated_ids.shape[1] if num_sequences > 1 else 0 # this is incorrect
            # Correction: The `generate` method, when num_return_sequences > 1 for a single input,
            # returns a tensor where each row is a full generated sequence (input + generated).
            # So, generated_ids will have a shape of (num_return_sequences, sequence_length).
            
            # Let's verify the shape of generated_ids directly.
            # If inputs.input_ids has shape (1, input_len) and num_return_sequences=4,
            # then generated_ids will have shape (4, output_len).

            # So, we just need to iterate through the rows of generated_ids.
            trimmed_id = generated_ids[i][input_id_length:]
            response = self.processor.decode(trimmed_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # output_texts.append(decoded_text)

            try:
                # For other cases, keep the original parsing logic
                reasoning = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                reasoning_content = reasoning.group(1).strip() if reasoning else ""
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)\n</answer>", response, re.DOTALL)
                
                answer_content = match.group(1).strip().lower().replace("species name: ", "").replace("make model: ", "").replace("species: ", "").replace("make and model: ", "")
                answer_content = clean_string(answer_content)  # Clean the answer content
                if ("barberton" in answer_content):
                    answer_content = answer_content.replace("barberton", "barbeton")
                
                if (len(answer_content) >= 100):
                    # Skip answers that are too long
                    # print(answer_content + " is too long, skipping...")
                    continue

                if answer_content not in curr_pred:
                    curr_pred[answer_content] = 1
                else:
                    curr_pred[answer_content] += 1

            except Exception as e:
                print(RED + "Error in processing response: " + RESET)
                print(RED + "Response: " + response + RESET)
        
        return curr_pred
    
    def get_mcq_output(self, inputs, generation_args):
        generated_ids = self.model.generate(**inputs, **generation_args)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        
        # print("\033[92m" + response + "\033[0m")

        try:

            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else response.strip()
            student_choice = extract_choice(student_answer)

            reasoning = re.search(r"<think>(.*?)</think>", response)
            reasoning_content = reasoning.group(1) if reasoning else ""
            
            return student_choice, reasoning_content
        
        except Exception as e:
            print(RED + "Error in processing response: " + RESET)
            print(RED + "Response: " + response + RESET)
            return None, None


def parse_args():
    parser = argparse.ArgumentParser(description="two step evaluation")
    parser.add_argument("--model_root", type=str, default="/app/saved_models/vrft/CUB_200_2011/")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--exp_name", type=str, default="Qwen2_5-VL-7B-Instruct_GRPO_cub_base_and_hard_mcq")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-400")
    parser.add_argument("--num_shot", type=int, default=0)
    parser.add_argument("--eval_type", type=str, default="baseline")
    parser.add_argument("--use_cat_list", type=str, default=False)
    parser.add_argument("--data_root", type=str, default="/data2/raja/")
    parser.add_argument("--dataset", type=str, default="CUB_200_2011")
    parser.add_argument("--split", type=str, default="new_val")
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()

def run(rank, world_size, args):

    MODEL_ROOT = args.model_root
    BASE_MODEL = args.base_model
    EXP_NAME = args.exp_name
    CHECKPOINT = args.checkpoint
    eval_type = args.eval_type
    use_cat_list = args.use_cat_list.lower() == "true"
    DATA_ROOT = args.data_root
    dataset = args.dataset
    split = args.split
    num_return_sequences = args.num_return_sequences
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    num_shot = args.num_shot

    model_path = os.path.join(MODEL_ROOT, f"{EXP_NAME}", CHECKPOINT)  # full path to the model"
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
    

    with open(category_file, 'r') as f:
        categories = f.read().splitlines()
    
    local_output_data = {}

    if "gemma-3" in model_base.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = Gemma3Processor.from_pretrained(model_base)
        # raise NotImplementedError("gemma-3 model loading not implemented yet.")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_base) 

    if "lora" in eval_type:

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_base,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )

        processor = AutoProcessor.from_pretrained(model_path)
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.bfloat16,
        )

        print("---------- Loading LoRA model from: ------------")

    

    model = model.to(torch.device(rank))
    model = model.eval()

    if (temperature == 0.0):
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_return_sequences": num_return_sequences,
            "use_cache": True,
            "temperature": None
        }
    else:
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }
    
    mcq_generation_args = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
    

    with open(data_json_path, 'r') as f:
        infer_data = json.load(f)
    
    random.seed(21)
    random.shuffle(infer_data)

    # infer_data = infer_data[-2:]

    print(GREEN + "Number of images in infer data: " + str(len(infer_data)) + RESET)
    
    split_length = math.ceil(len(infer_data)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = infer_data[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))
    
    print(YELLOW + f"generation args: {generation_args}" + RESET)

    two_step_inference = TwoStepInference(model, processor)

    for item in tqdm(split_images, total=len(split_images), desc=f"Rank {rank} Processing"):
        image_path = item['image_path']
        image_label = item['solution']

        prompt = item['problem']
        image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
        image_label = clean_string(image_label)  # Clean the image label
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/fgvc_aircraft/",
                        DATA_ROOT)
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", 
                        DATA_ROOT)

        instruction, answer_format, data_type = PROMPTS[dataset]["instruction"], PROMPTS[dataset]["answer_format"], PROMPTS[dataset]["data_name"]

        if use_cat_list:
            question = (
            f"This is an image containing a {data_type}. {instruction}\n"
            f"the {answer_format} of the {data_type} strictly belongs to below category list {categories}.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            f"<think> ... </think> <answer> {answer_format} </answer>\n"
            +
            "Please strictly follow the format."
            )
        else:
            question = prompt
    
        query = "<image>\n"+question
        
        if ("gemma-3" in model_base.lower()):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            prompts_text = processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            image_raw = Image.open(image_path).convert("RGB").resize((896, 896))
            inputs = processor(
                text=prompts_text,
                images=image_raw,
                return_tensors="pt",
            )

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path}
                    ] + [{"type": "text", "text": query}],
                }
            ]
            
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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

        # Inference: Get the pass@20 output
        pass_at_20_output = two_step_inference.get_pass_at_20(inputs, generation_args)

        top5 = post_process_passk(pass_at_20_output)
        top5_keys = list(top5.keys())
        random.shuffle(top5_keys)  # Shuffle the options to avoid bias

        if len(top5_keys) == 0:
            print("step1 failed, skipping image: " + image_path)
            continue
        # MCQ inference now

        letters = ['A', 'B', 'C', 'D', 'E']
        options = "\n".join([f"{letters[i]}. {option}" for i, option in enumerate(top5_keys)])

        mcq_prompt = f""" This is an image containing a {data_type}. Please find the most likely {data_type} in the image from the below options.
{options}
Please output the letter corresponding to the correct {data_type} name.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:
<think> ... </think> <answer>option letter</answer>
Please strictly follow the format. """
        
        query = "<image>\n"+mcq_prompt
        # print(RED+query+RESET)
        
        if ("gemma-3" in model_base.lower()):
            mcq_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": mcq_prompt}
                    ]
                }
            ]

            mcq_inputs = processor.apply_chat_template(
                    mcq_messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", do_pan_and_scan=True
            )
        else:
            mcq_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path}
                    ] + [{"type": "text", "text": query}],
                }
            ]
            
            # Preparation for inference
            text = processor.apply_chat_template(
                mcq_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(mcq_messages)

            mcq_inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        

        mcq_inputs = mcq_inputs.to(model.device)

        mcq_output, mcq_reasoning = two_step_inference.get_mcq_output(mcq_inputs, mcq_generation_args)

        try:
            predicted_category = top5_keys[letters.index(mcq_output)] if mcq_output in letters else None
        except Exception as e:
            print(RED + f"Error: mcq_output: {mcq_output} \n mcq_input: {options}" + RESET)
            predicted_category = None

        image_id = image_path.split("/")[-1].split(".")[0]  # Extract image ID from the path

        ## add the groundtruth to the output data
        local_output_data[image_id] = {
            "groundtruth": image_label,
            "step1_output": pass_at_20_output,
            "options": top5_keys,
            "step2_output": mcq_output,
            "prediction": predicted_category,
            "step2_reasoning": mcq_reasoning,
        }
        
    return [local_output_data]

def main():

    args = parse_args()
    MODEL_ROOT = args.model_root
    dataset = args.dataset
    eval_type = args.eval_type
    EXP_NAME = args.exp_name
    CHECKPOINT = args.checkpoint
    split = args.split
    k = args.num_return_sequences


    output_path = f"./output/{dataset}/{eval_type}/"
    model_path = os.path.join(MODEL_ROOT, f"{EXP_NAME}", CHECKPOINT)

    if EXP_NAME != "baseline":
        model_name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1] # use checkpoint name
    else:
        model_name = args.base_model.split("/")[-1]  # model name

    output_file = f"{model_name}_{split}_pass{k}.json"  # output file name

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file_path = os.path.join(output_path, output_file)
    print(GREEN + "output path" + output_file_path + RESET)


    multiprocess = torch.cuda.device_count() >= 1
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, args=args)
            result_lists = pool.map(func, range(world_size))

        output_data = {}
        for i in range(world_size):
            output_data.update(result_lists[i][0])  # merge local output data
        
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Results saved to {output_file_path}")
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()