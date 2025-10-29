import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify

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

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                ground_truth = ground_truth.replace(' ','').replace('_','').lower()
                student_answer = student_answer.replace(' ','').replace('_','').lower()

                # Compare the extracted answers
                # if ground_truth in student_answer or student_answer in ground_truth:
                if ground_truth in student_answer:
                    # print(f"using the correct reward")
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward_v2(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = (
        r"<describe>.*?</describe>\s*"
        r"<think>.*?</think>\s*"
        r"<rethink>.*?</rethink>\s*"
        r"<answer>.*?</answer>"
    )
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def mcq_reward(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        # print(f"content: {content}")
        # print(f"sol: {sol}")
        # For multiple choice, extract and compare choices
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        has_choices = extract_choice(ground_truth)
        correct_choice = has_choices.upper() if has_choices else sol.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        student_choice = extract_choice(student_answer)
        if student_choice:
            reward = 1.0 if student_choice == correct_choice else 0.0
        else:
            reward = 0.0
        
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    return rewards