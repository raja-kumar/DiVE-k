import json
import random
import re
from tqdm import tqdm
from utils import pred_class_to_idx, clean_topk, clean_string, post_process_passk
from topk_gen import generate_topk_predictions, parse_args as topk_parse_args
import tempfile
import os
import argparse

LETTERS = [chr(i) for i in range(ord('A'), ord('Z')+1)]

def sample_random_options(categories, gt_cat_name, num_options=4):
    """Sample random options, excluding the ground truth."""
    gt_clean = clean_string(gt_cat_name).lower()
    filtered_keys = [k for k in categories if clean_string(k).lower() != gt_clean]
    return random.sample(filtered_keys, min(num_options, len(filtered_keys)))

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

def build_mcq_prompt(data_name, options_str):
    return (
        f"This is an image containing a {data_name}. Please find the most likely {data_name} in the image from the below options.\n"
        f"{options_str}\n"
        f"Please output the letter corresponding to the correct {data_name} name.\n"
        f"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n"
        f"<think> ... </think> <answer>option letter</answer>\n"
        f"Please strictly follow the format."
    )

def generate_mcq_data(top5_pred_file, json_data_path, data_name):
    with open(top5_pred_file, 'r') as f:
        top5_data = json.load(f)
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    print(f"Top 5 data length: {len(top5_data)}; JSON data length: {len(json_data)}")

    dataset, skipped = [], 0
    for item in tqdm(json_data):
        image_path = item["image_path"]
        gt_cat_name = re.search("<answer>(.*?)</answer>", item["solution"]).group(1)
        image_id = image_path.split("/")[-1].split(".")[0]

        curr_data = top5_data.get(image_id)
        if not curr_data:
            continue

        gpt_preds = curr_data["gpt_pred"][:5]
        gpt_labels = curr_data["pred_labels"][:5]
        gt_label = curr_data["gt_label"]

        # Make sure gt_cat_name is an option
        if gt_label == -1 or (gt_label not in gpt_labels) or (len(gpt_labels) == 0):
            gpt_preds[-1] = gt_cat_name
        
        random.shuffle(gpt_preds)
        options_str = format_options(gpt_preds)
        answer_letter = find_mcq_answer(gpt_preds, gt_cat_name)

        if answer_letter == -1:
            skipped += 1
            continue

        prompt = build_mcq_prompt(data_name, options_str)
        dataset.append({
            "image_path": image_path,
            "problem": prompt,
            "solution": f"<answer>{answer_letter}</answer>"
        })
    random.shuffle(dataset)
    print(f"Total count of skipped images: {skipped}")
    return dataset

def generate_mcq_qwen(top5_pred_file, json_data_path, data_name, cat_2_idx_path):
    with open(top5_pred_file, 'r') as f:
        top5_data = json.load(f)
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    with open(cat_2_idx_path, 'r') as f:
        class_to_idx_raw = json.load(f)

    # Normalize class names
    class_to_idx = {clean_string(k): v for k, v in class_to_idx_raw.items()}
    categories = list(class_to_idx.keys())

    print(f"Top 5 data length: {len(top5_data)}; JSON data length: {len(json_data)}")
    random.shuffle(json_data)
    dataset = []
    skipped = top5_count = one_answer_count = 0

    for item in tqdm(json_data):
        image_path = item["image_path"]
        gt_cat_name = clean_string(re.search("<answer>(.*?)</answer>", item["solution"]).group(1))
        image_id = image_path.split("/")[-1].split(".")[0]

        try:
            curr_data = top5_data[image_id]
            gpt_preds = list(post_process_passk(curr_data["predictions"]).keys())
        except Exception:
            gpt_preds = sample_random_options(categories, gt_cat_name, 2)
        
        gpt_preds = clean_topk(gpt_preds, class_to_idx)
        gt_label = pred_class_to_idx([gt_cat_name], class_to_idx)[0]
        gpt_labels = pred_class_to_idx(gpt_preds, class_to_idx)

        # Add random options if too few
        if len(gpt_preds) <= 1:
            if gt_cat_name in gpt_preds:
                skipped += 1
                continue
            one_answer_count += 1
            random_options = sample_random_options(categories, gt_cat_name, 2)
            gpt_preds += random_options

        # Ensure gt_cat_name is in options
        if gt_label == -1 or (gt_label not in gpt_labels):
            top5_count += 1
            if len(gpt_preds) < 5:
                gpt_preds.append(gt_cat_name)
            else:
                gpt_preds[-1] = gt_cat_name
        
        random.shuffle(gpt_preds)
        options_str = format_options(gpt_preds)
        answer_letter = find_mcq_answer(gpt_preds, gt_cat_name)

        if answer_letter == -1:
            skipped += 1
            continue

        prompt = build_mcq_prompt(data_name, options_str)
        dataset.append({
            "image_path": image_path,
            "problem": prompt,
            "solution": f"<answer>{answer_letter}</answer>"
        })

    random.shuffle(dataset)
    print(f"Total count of skipped images: {skipped}")
    print(f"Total count of top5 predictions: {top5_count}")
    print(f"Total count of one answer predictions: {one_answer_count}")
    return dataset

def generate_mcq_random(json_data_path, data_name, category_names_path):
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    with open(category_names_path, 'r') as f:
        categories = [clean_string(cat) for cat in f.read().splitlines()]
    print(f"JSON data length: {len(json_data)}")
    random.shuffle(json_data)
    dataset, skipped = [], 0

    for item in tqdm(json_data):
        image_path = item["image_path"]
        gt_cat_name = clean_string(re.search("<answer>(.*?)</answer>", item["solution"]).group(1))
        random_options = sample_random_options(categories, gt_cat_name, num_options=4)
        options = random_options + [gt_cat_name]
        random.shuffle(options)
        options_str = format_options(options)
        answer_letter = find_mcq_answer(options, gt_cat_name)

        if answer_letter == -1 or len(options) <= 1:
            skipped += 1
            continue

        prompt = build_mcq_prompt(data_name, options_str)
        dataset.append({
            "image_path": image_path,
            "problem": prompt,
            "solution": f"<answer>{answer_letter}</answer>"
        })

    random.shuffle(dataset)
    print(f"Total count of skipped images: {skipped}")
    return dataset

def generate_mcq_text_embedding(json_data_path, data_name, options_file):
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)
    with open(options_file, 'r') as f:
        text_embedding_options = json.load(f)
    random.shuffle(json_data)
    dataset, skipped = [], 0

    for item in tqdm(json_data):
        image_path = item["image_path"]
        gt_cat_name = clean_string(re.search("<answer>(.*?)</answer>", item["solution"]).group(1))
        category_key = str(int(image_path.split("/")[-2].split(".")[0])) # Bird dataset specific
        options = text_embedding_options[category_key]
        random.shuffle(options)
        options_str = format_options(options)
        answer_letter = find_mcq_answer(options, gt_cat_name)

        if answer_letter == -1 or len(options) <= 1:
            skipped += 1
            continue

        prompt = build_mcq_prompt(data_name, options_str)
        dataset.append({
            "image_path": image_path,
            "problem": prompt,
            "solution": f"<answer>{answer_letter}</answer>"
        })

    random.shuffle(dataset)
    print(f"Total count of skipped images: {skipped}")
    return dataset

def generate_mcq_qwen_with_topk(json_data_path, data_name, cat_2_idx_path, topk_args, intermediate_output_dir=None):
    """
    Generate MCQ data using Qwen model, with automatic top-k generation.
    Saves intermediate top-k predictions to a permanent file.
    """
    # Generate top-k predictions first
    print("Generating top-k predictions...")
    
    # Create intermediate output directory if not provided
    if intermediate_output_dir is None:
        intermediate_output_dir = "./intermediate_topk_results"
    
    os.makedirs(intermediate_output_dir, exist_ok=True)
    
    # Generate a descriptive filename for the intermediate results
    base_filename = f"{topk_args.dataset}_{topk_args.split}_{topk_args.base_model.split('/')[-1]}"
    base_filename += f"_topk_{topk_args.num_return_sequences}_temp_{topk_args.temperature}"
    intermediate_file = os.path.join(intermediate_output_dir, f"{base_filename}.json")
    
    print(f"Intermediate top-k predictions will be saved to: {intermediate_file}")
    
    # Generate top-k predictions and save to permanent file
    actual_topk_file = generate_topk_predictions(topk_args, intermediate_file)
    
    # Now generate MCQ data using the top-k predictions
    dataset = generate_mcq_qwen(actual_topk_file, json_data_path, data_name, cat_2_idx_path)
    
    print(f"Top-k predictions saved at: {actual_topk_file}")
    return dataset, actual_topk_file

def save_dataset(dataset, output_path):
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to {output_path}")
    print(f"Generated MCQ dataset with {len(dataset)} items.")

class TopkArgs:
    """Arguments class for top-k generation that can be pickled."""
    def __init__(self, mcq_args):
        self.model_root = "/app/saved_models/vrft/CUB_200_2011/"
        self.base_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.exp_name = "baseline"
        self.checkpoint = "checkpoint-400"
        self.num_shot = mcq_args.num_shot
        self.eval_type = "baseline"
        self.use_cat_list = True
        self.data_root = mcq_args.data_root
        self.dataset = mcq_args.data
        self.split = f"{mcq_args.split}_{mcq_args.phase}"
        self.num_return_sequences = 20
        self.temperature = 1.0
        self.max_new_tokens = 512

def create_topk_args(mcq_args):
    """
    Create topk_gen arguments from mcq arguments.
    """
    return TopkArgs(mcq_args)


def main():
    parser = argparse.ArgumentParser(description="Generate MCQ datasets for VLM evaluation")
    parser.add_argument("--data_root", type=str, default="/data2/raja/")
    parser.add_argument("--data", type=str, default="oxford_flowers")
    parser.add_argument("--split", type=str, default="base")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--data_name", type=str, default="flowers")
    parser.add_argument("--mcq_type", type=str, choices=["qwen", "random", "gemini", "text_embedding"], default="qwen")
    parser.add_argument("--num_shot", type=int, default=0)
    parser.add_argument("--top5_pred_file", type=str, default="", help="Pre-existing top-k prediction file (optional)")
    
    # Additional arguments for top-k generation
    parser.add_argument("--force_regenerate_topk", action="store_true", help="Force regeneration of top-k predictions even if file exists")
    parser.add_argument("--intermediate_dir", type=str, default="./intermediate_topk_results", help="Directory to save intermediate top-k prediction files")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep intermediate top-k prediction files after MCQ generation")
    
    args = parser.parse_args()
    
    # Set up paths
    if args.num_shot > 0:
        json_data_path = f"{args.data_root}/{args.data}/fewshot/{args.num_shot}_shots_all_train.json"
        output_path = f"{args.data_root}/{args.data}/fewshot/{args.num_shot}_shots_all_train_mcq.json"
    else:
        json_data_path = f"{args.data_root}/{args.data}/zero_shot/subsample_{args.split}_{args.phase}.json"
        output_path = f"{args.data_root}/{args.data}/zero_shot/subsample_{args.split}_{args.phase}_all_mcq.json"
    
    class_to_idx_path = f"{args.data_root}/{args.data}/class_2_idx.json"
    
    if args.mcq_type == "qwen":
        print("Using Qwen output format")
        
        # Check if top-k prediction file is provided and exists
        if args.top5_pred_file and os.path.exists(args.top5_pred_file) and not args.force_regenerate_topk:
            print(f"Using existing top-k prediction file: {args.top5_pred_file}")
            dataset = generate_mcq_qwen(args.top5_pred_file, json_data_path, args.data_name, class_to_idx_path)
            intermediate_file = args.top5_pred_file  # For reporting purposes
        else:
            print("Generating top-k predictions automatically...")
            topk_args = create_topk_args(args)
            dataset, intermediate_file = generate_mcq_qwen_with_topk(
                json_data_path, args.data_name, class_to_idx_path, topk_args, args.intermediate_dir
            )
        
        output_path = f"{args.data_root}/{args.data}/qwen_mcq/subsample_{args.split}_{args.phase}_pass_20_all_mcq.json"
        
    elif args.mcq_type == "gemini":
        top5_pred_file = f"/home/raja/OVOD/git_files/VLM-COT/outputs/{args.data}/{args.data}_step1_baseline_{args.split}_{args.phase}_gemini-2.5-flash-lite-preview-06-17_cat_True.json"
        dataset = generate_mcq_data(top5_pred_file, json_data_path, args.data_name)
        intermediate_file = top5_pred_file
        
    elif args.mcq_type == "text_embedding":
        output_path = f"{args.data_root}/{args.data}/text_emb_mcq/subsample_{args.split}_{args.phase}_mcq.json"
        options_file = f"{args.data_root}/CUB_200_2011/text_similar_categories.json"
        dataset = generate_mcq_text_embedding(json_data_path, args.data_name, options_file)
        intermediate_file = None  # No intermediate file for this type
        
    elif args.mcq_type == "random":
        print("Using random options")
        category_name_path = f"{args.data_root}/{args.data}/zero_shot/base_categories.txt"
        dataset = generate_mcq_random(json_data_path, args.data_name, category_name_path)
        output_path = f"{args.data_root}/{args.data}/random_mcq/subsample_{args.split}_{args.phase}_mcq.json"
        intermediate_file = None  # No intermediate file for this type
        
    else:
        raise ValueError("Unknown MCQ type")

    save_dataset(dataset, output_path)
    
    # Report intermediate file location if it exists
    if intermediate_file:
        print(f"Intermediate top-k predictions available at: {intermediate_file}")
        
        # Option to clean up intermediate file if not keeping it
        if not args.keep_intermediate and args.mcq_type == "qwen" and args.top5_pred_file == "":
            print("Note: Use --keep_intermediate flag to preserve the intermediate top-k prediction file for future use.")
            print("Or specify it directly next time with --top5_pred_file argument to skip regeneration.")

if __name__ == "__main__":
    main()