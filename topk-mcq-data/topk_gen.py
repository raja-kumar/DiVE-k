import os
import re
import json
import torch
import random
import logging
import argparse
import math
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import functools

from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2VLForConditionalGeneration
    # Gemma3ForConditionalGeneration, Gemma3Processor  # Uncomment if using
)
from qwen_vl_utils import process_vision_info
from prompts import PROMPTS

# ANSI color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Logging setup
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Top-K Accuracy Evaluation")
    parser.add_argument("--model_root", type=str, default="/app/saved_models/vrft/CUB_200_2011/")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--exp_name", type=str, default="Qwen2_5-VL-7B-Instruct_GRPO_cub_base_and_hard_mcq")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-400")
    parser.add_argument("--num_shot", type=int, default=0)
    parser.add_argument("--eval_type", type=str, default="baseline")
    parser.add_argument("--use_cat_list", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--data_root", type=str, default="/data2/raja/")
    parser.add_argument("--dataset", type=str, default="CUB_200_2011")
    parser.add_argument("--split", type=str, default="base_train")
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()

def clean_string(text):
    """Cleans the input text by removing unwanted characters and formatting."""
    text = text.replace("'s", "")
    text = re.sub(r'[^a-zA-Z0-9-]', ' ', text)
    return text.strip().lower()

def get_paths_and_filenames(args):
    model_path = os.path.join(args.model_root, args.exp_name, args.checkpoint)
    if args.exp_name == "baseline":
        model_path = args.base_model
    if args.num_shot > 0:
        data_json_path = f"{args.data_root}/{args.dataset}/fewshot/{args.num_shot}_shots_all_train.json"
    else:
        data_json_path = f"{args.data_root}/{args.dataset}/zero_shot/subsample_{args.split}.json"
    output_path = f"./output/{args.dataset}/{args.eval_type}/"
    if "checkpoint" in model_path:
        model_name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1]
    else:
        model_name = model_path.split("/")[-1]
    data_name = data_json_path.split("/")[-1].split(".")[0]
    output_file = f"{model_name}_{data_name}_{args.use_cat_list}_{args.num_return_sequences}_{args.temperature}.json"
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, output_file)
    return model_path, data_json_path, output_file_path

def load_categories(args):
    split_name = args.split.split("_")[0]
    if args.num_shot > 0:
        category_file = f"{args.data_root}/{args.dataset}/fewshot/all_categories.txt"
    else:
        category_file = f"{args.data_root}/{args.dataset}/zero_shot/{split_name}_categories.txt"
    with open(category_file, 'r') as f:
        categories = f.read().splitlines()
    return categories

def prepare_model_and_processor(model_base, model_path):
    if "Qwen2.5" in model_base:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_base) 
    elif "Phi-3.5" in model_base:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            _attn_implementation='flash_attention_2'    
        )
        processor = AutoProcessor.from_pretrained(model_base, 
            trust_remote_code=True, 
            num_crops=16,
        )
        print(GREEN + "Using Phi-3.5 model" + RESET)
    elif "gemma-3" in model_base:
        # model = Gemma3ForConditionalGeneration.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="cpu",
        # )
        # processor = Gemma3Processor.from_pretrained(model_base)
        raise NotImplementedError("Gemma-3 support not implemented in this script.")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_base)
    return model, processor

def create_prompt(item, args, categories):
    dataset = args.dataset
    prompt = item['problem']
    temp, answer_format, data_name = PROMPTS[dataset]["instruction"], PROMPTS[dataset]["answer_format"], PROMPTS[dataset]["data_name"]
    if args.use_cat_list:
        question = (
            f"This is an image containing a {data_name}. {temp}\n"
            f"The {answer_format} of the {data_name} strictly belongs to below category list {categories}.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            f"<think> ... </think> <answer> {answer_format} </answer>\n"
            "Please strictly follow the format."
        )
    else:
        question = prompt
    return question

def prepare_inputs(item, question, model_base, processor, DATA_ROOT, dataset):
    image_path = item['image_path']
    if dataset == "fgvc_aircraft":
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/fgvc_aircraft/", DATA_ROOT)
    else:
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", DATA_ROOT)
    if "Phi-3.5" in model_base:
        images = [Image.open(image_path)]
        query = "<|image_1|>\n" + question
        messages = [{"role": "user", "content": query}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt")
    elif "gemma-3" in model_base:
        raise NotImplementedError("Gemma-3 support not implemented in this script.")
    else:
        query = "<image>\n" + question
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    return inputs

def get_generation_args(temperature, max_new_tokens, num_return_sequences):
    if temperature == 0.0:
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_return_sequences": num_return_sequences,
            "use_cache": True,
            "temperature": None
        }
    else:
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }

def process_generated_output(
    generated_ids, 
    input_id_length, 
    num_sequences, 
    processor, 
    answer_format, 
    image_label
):
    curr_pred = {}
    for i in range(num_sequences):
        trimmed_id = generated_ids[i][input_id_length:]
        response = processor.decode(trimmed_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        try:
            reasoning = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            reasoning_content = reasoning.group(1).strip() if reasoning else ""
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if not match:
                match = re.search(r"<answer>\n(.*?)</answer>", response, re.DOTALL)
            if not match:
                match = re.search(r"<answer>\n(.*?)\n</answer>", response, re.DOTALL)
            answer_content = match.group(1).strip().lower().replace(f"{answer_format}: ", "")
            answer_content = clean_string(answer_content)
            if answer_content not in curr_pred:
                curr_pred[answer_content] = 1
            else:
                curr_pred[answer_content] += 1
        except Exception as e:
            print(RED + f"Error in processing response: {e}" + RESET)
            print(RED + "Response: " + response + RESET)
    return curr_pred

def run(rank, world_size, args, shared_args):
    random.seed(21)
    torch.manual_seed(1234)
    model, processor = prepare_model_and_processor(shared_args['model_base'], shared_args['model_path'])
    model = model.to(torch.device(rank))
    model = model.eval()
    with open(shared_args['data_json_path'], 'r') as f:
        infer_data = json.load(f)
    random.shuffle(infer_data)
    split_length = math.ceil(len(infer_data) / world_size)
    split_images = infer_data[int(rank * split_length): int((rank + 1) * split_length)]
    logger.info(f"Rank {rank}: Processing {len(split_images)} images")
    if args.use_cat_list:
        categories = shared_args['categories']
    else:
        categories = None
    generation_args = get_generation_args(
        args.temperature, args.max_new_tokens, args.num_return_sequences
    )
    print(YELLOW + f"generation args: {generation_args}" + RESET)
    local_output_data = {}
    for item in tqdm(split_images, total=len(split_images), desc=f"Rank {rank} Processing"):
        image_id = item['image_path'].split("/")[-1].split(".")[0]
        image_label = item['solution']
        image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
        image_label = clean_string(image_label)
        question = create_prompt(item, args, categories)
        try:
            inputs = prepare_inputs(item, question, shared_args['model_base'], processor, args.data_root, args.dataset)
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, **generation_args)
            input_id_length = inputs.input_ids.shape[1]
            num_sequences = generation_args["num_return_sequences"]
            answer_format = PROMPTS[args.dataset]["answer_format"]
            curr_pred = process_generated_output(
                generated_ids, input_id_length, num_sequences, processor, answer_format, image_label
            )
            local_output_data[image_id] = {
                "groundtruth": image_label,
                "predictions": curr_pred,
            }
        except Exception as e:
            print(RED + "Error during processing: " + str(e) + RESET)
            print(RED + "Skipping image: " + item['image_path'] + RESET)
            continue
    return [local_output_data]

def main():
    args = parse_args()
    generate_topk_predictions(args)
    # random.seed(21)
    # torch.manual_seed(1234)
    # model_path, data_json_path, output_file_path = get_paths_and_filenames(args)
    # shared_args = {
    #     "model_base": args.base_model,
    #     "model_path": model_path,
    #     "data_json_path": data_json_path,
    #     "categories": load_categories(args) if args.use_cat_list else None
    # }
    # output_data = {}
    # use_multi = torch.cuda.device_count() >= 1
    # set_start_method('spawn', force=True)
    # if use_multi:
    #     logger.info('Started generation')
    #     n_gpus = torch.cuda.device_count()
    #     world_size = n_gpus
    #     with Pool(world_size) as pool:
    #         func = functools.partial(run, world_size=world_size, args=args, shared_args=shared_args)
    #         result_lists = pool.map(func, range(world_size))
    #     logger.info('Finished generation')
    #     for i in range(world_size):
    #         output_data.update(result_lists[i][0])
    # else:
    #     logger.info("Not enough GPUs")
    # print(GREEN + "output path: " + output_file_path + RESET)
    # with open(output_file_path, 'w') as f:
    #     json.dump(output_data, f, indent=4)
    # logger.info(f"Output saved to {output_file_path}")

def generate_topk_predictions(args, output_file_path=None):
    """
    Generate top-k predictions for the given arguments.
    If output_file_path is None, creates a temporary file.
    Returns the path to the output file.
    """
    random.seed(21)
    torch.manual_seed(1234)
    
    model_path, data_json_path, default_output_path = get_paths_and_filenames(args)
    
    # Use provided output path or default
    final_output_path = output_file_path or default_output_path
    
    shared_args = {
        "model_base": args.base_model,
        "model_path": model_path,
        "data_json_path": data_json_path,
        "categories": load_categories(args) if args.use_cat_list else None
    }
    
    output_data = {}
    use_multi = torch.cuda.device_count() >= 1
    set_start_method('spawn', force=True)
    
    if use_multi:
        logger.info('Started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, args=args, shared_args=shared_args)
            result_lists = pool.map(func, range(world_size))
        logger.info('Finished generation')
        for i in range(world_size):
            output_data.update(result_lists[i][0])
    else:
        logger.info("Not enough GPUs - using single process")
        result = run(0, 1, args, shared_args)
        output_data.update(result[0])
    
    # Save to file
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    with open(final_output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    logger.info(f"Top-k predictions saved to {final_output_path}")
    return final_output_path

if __name__ == "__main__":
    main()