import json
import re

def get_cat_name_from_json(json_file_path):
    """
    Get category names from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing category names.
        
    Returns:
        list: A list of category names.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    categories = [data[key] for key in data]

    print(f"Number of categories: {len(categories)}")
    print(categories)  # Print the category names
    return categories

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


    

def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]