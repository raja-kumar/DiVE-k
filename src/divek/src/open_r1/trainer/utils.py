import re
import json
import random
from open_r1.trainer.prompts import PROMPTS, prompts

LETTERS = [chr(i) for i in range(ord('A'), ord('Z')+1)]


def clean_string(text):
    """
    Cleans the input text by removing unwanted characters and formatting.
    """
    text = text.replace("'s", "")
    text = re.sub(r'[^a-zA-Z0-9-]', ' ', text)
    text = text.strip().lower()
    
    return text

def process_generated_output(
    generated_ids, 
    input_id_length, 
    num_sequences, 
    processor, 
    answer_format
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
            print(f"Error in processing response: {e}")
            # print("Response: " + response)
    
    return curr_pred

def clean_topk(topk_list, class_to_idx):
    """
    Cleans the top-k predictions by removing invalid entries and ensuring they are unique.
    """
    cleaned_topk = set()

    for item in topk_list:
        item = item.replace("'s", "")
        item = re.sub(r'[^a-zA-Z0-9-]', ' ', item)
        item = item.strip().lower()
        if (len(item) > 120):
            continue

        if ("barberton" in item):
            item = item.replace("barberton", "barbeton")

        if item in class_to_idx:
            cleaned_topk.add(item)
    
    return list(cleaned_topk)

def post_process_passk(passk_output):
    sorted_dict_desc = sorted(passk_output.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_dict_desc[:min(5, len(sorted_dict_desc))])

def sample_random_options(categories, gt_cat_name, num_options=4):
    """Sample random options, excluding the ground truth."""
    gt_clean = clean_string(gt_cat_name).lower()
    filtered_keys = [k for k in categories if clean_string(k).lower() != gt_clean]
    return random.sample(filtered_keys, min(num_options, len(filtered_keys)))

def build_mcq_prompt(data_name, options_str):
    return (
        f"This is an image containing a {data_name}. Please find the most likely {data_name} in the image from the below options.\n"
        f"{options_str}\n"
        f"Please output the letter corresponding to the correct {data_name} name.\n"
        f"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n"
        f"<think> ... </think> <answer>option letter</answer>\n"
        f"Please strictly follow the format."
    )

def pred_class_to_idx(cat_list, class_to_idx):
    idx_list = []
    for cat in cat_list:
        cat = cat.replace("'s", "")
        cat = re.sub(r'[^a-zA-Z0-9-]', ' ', cat)
        cat = cat.strip().lower()

        if ("barberton" in cat):
            cat = cat.replace("barberton", "barbeton")
        
        if cat in class_to_idx:
            idx_list.append(class_to_idx[cat])
        else:
            idx_list.append(-1)
    
    return idx_list

def format_options(options):
    """Format options as multiple choice letters."""
    return "\n".join([f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)])

def find_mcq_answer(options, gt_cat_name):
    """Find the correct MCQ letter for the ground-truth category."""
    gt_norm = gt_cat_name.lower().replace("-", " ")
    for i, option in enumerate(options):
        if option.lower().replace("-", " ") == gt_norm:
            return LETTERS[i]
    return -1

def create_prompt(categories, dataset):
    temp, answer_format, data_name = PROMPTS[dataset]["instruction"], PROMPTS[dataset]["answer_format"], PROMPTS[dataset]["data_name"]
    data_name = PROMPTS[dataset]["data_name"]
    question = (
        f"This is an image containing a {data_name}. {temp}\n"
        f"The {answer_format} of the {data_name} strictly belongs to below category list {categories}.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        f"<think> ... </think> <answer> {answer_format} </answer>\n"
        "Please strictly follow the format."
    )

    return question