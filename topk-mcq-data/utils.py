import re
import os

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

def clean_string(text):
    """
    Cleans the input text by removing unwanted characters and formatting.
    """
    text = text.replace("'s", "")
    text = re.sub(r'[^a-zA-Z0-9-]', ' ', text)
    text = text.strip().lower()
    
    return text

def post_process_passk(passk_output):
    sorted_dict_desc = sorted(passk_output.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_dict_desc[:min(5, len(sorted_dict_desc))])

def find_existing_topk_file(args, intermediate_dir):
    """
    Look for existing top-k prediction files that match the current configuration.
    """
    if not os.path.exists(intermediate_dir):
        return None
    
    # Create expected filename pattern
    base_filename = f"{args.data_folder}_{args.split}_{args.phase}_Qwen2.5-VL-7B-Instruct"
    base_filename += f"_topk_20_temp_1.0.json"
    
    expected_file = os.path.join(intermediate_dir, base_filename)
    
    if os.path.exists(expected_file):
        print(f"Found existing top-k prediction file: {expected_file}")
        return expected_file
    
    return None