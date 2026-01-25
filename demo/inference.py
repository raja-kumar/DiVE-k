"""
Sample inference code for DiVE-k-QWEN2.5-7B-CUB model
Author: Based on evaluation code from raja-kumar
Model: raja-kumar/DiVE-k-QWEN2.5-7B-CUB

This script demonstrates how to use the fine-tuned model for two-step inference:
1. Generate top-k predictions (pass@k)
2. Refine predictions using MCQ-style selection
"""

import re
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Set random seed for reproducibility
torch.manual_seed(1234)

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def clean_string(s):
    """Clean and normalize string output"""
    s = s.strip().lower()
    # Remove common prefixes
    s = s.replace("species name: ", "").replace("species: ", "")
    s = s.replace("make model: ", "").replace("make and model: ", "")
    # Fix known typos
    if "barberton" in s:
        s = s.replace("barberton", "barbeton")
    return s


def extract_choice(text):
    """Extract choice letter (A, B, C, D, E) from MCQ answer"""
    text = text.strip().upper()
    # Look for single letter options
    match = re.search(r'\b([A-E])\b', text)
    if match:
        return match.group(1)
    return text[0] if text and text[0] in 'ABCDE' else None


class DiVEInference:
    """Two-step inference pipeline for fine-grained visual recognition"""
    
    def __init__(self, model_name="raja-kumar/DiVE-k-QWEN2.5-7B-CUB", device="cuda"):
        """
        Initialize the model and processor
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda' or 'cpu')
        """
        print(f"{YELLOW}Loading model from {model_name}...{RESET}")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        self.model.eval()
        
        # Load processor from base model
        base_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(base_model)
        
        print(f"{GREEN}Model loaded successfully!{RESET}")
    
    def get_pass_at_k(self, image_path, question, num_sequences=20, temperature=1.0, max_tokens=512):
        """
        Step 1: Generate top-k predictions using sampling
        
        Args:
            image_path: Path to input image
            question: Text prompt/question
            num_sequences: Number of predictions to generate (k)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary of predictions with their frequencies
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"<image>\n{question}"}
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generation arguments
        if temperature == 0.0:
            gen_args = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "num_return_sequences": num_sequences,
                "use_cache": True,
                "temperature": None
            }
        else:
            gen_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True,
                "num_return_sequences": num_sequences,
                "repetition_penalty": 1.1,
                "use_cache": True,
            }
        
        # Generate predictions
        print(f"{YELLOW}Generating {num_sequences} predictions...{RESET}")
        generated_ids = self.model.generate(**inputs, **gen_args)
        
        input_len = inputs.input_ids.shape[1]
        predictions = {}
        
        # Process each generated sequence
        for i in range(num_sequences):
            trimmed_id = generated_ids[i][input_len:]
            response = self.processor.decode(
                trimmed_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            try:
                # Extract answer from tags
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)</answer>", response, re.DOTALL)
                
                if match:
                    answer = clean_string(match.group(1).strip())
                    
                    # Skip very long answers
                    if len(answer) >= 100:
                        continue
                    
                    # Count frequency
                    if answer not in predictions:
                        predictions[answer] = 1
                    else:
                        predictions[answer] += 1
            except Exception as e:
                print(f"{RED}Error processing response: {e}{RESET}")
                continue
        
        return predictions
    
    def get_top_k_candidates(self, predictions, k=5):
        """
        Filter and sort predictions to get top-k candidates
        
        Args:
            predictions: Dictionary of predictions with frequencies
            k: Number of top candidates to return
            
        Returns:
            List of top-k prediction strings
        """
        # Sort by frequency (descending)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        # Return top-k keys
        return [pred[0] for pred in sorted_preds[:k]]
    
    def refine_with_mcq(self, image_path, candidates, data_type="bird"):
        """
        Step 2: Refine predictions using MCQ-style selection
        
        Args:
            image_path: Path to input image
            candidates: List of candidate predictions
            data_type: Type of object (e.g., "bird", "aircraft")
            
        Returns:
            Tuple of (selected_answer, reasoning)
        """
        if not candidates:
            return None, None
        
        # Create MCQ options
        letters = ['A', 'B', 'C', 'D', 'E']
        options = "\n".join([f"{letters[i]}. {option}" for i, option in enumerate(candidates)])
        
        mcq_prompt = f"""This is an image containing a {data_type}. Please find the most likely {data_type} in the image from the below options.
{options}
Please output the letter corresponding to the correct {data_type} name.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:
<think> ... </think> <answer>option letter</answer>
Please strictly follow the format."""
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"<image>\n{mcq_prompt}"}
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate MCQ answer
        print(f"{YELLOW}Refining with MCQ selection...{RESET}")
        gen_args = {
            "max_new_tokens": 512,
            "use_cache": True,
        }
        generated_ids = self.model.generate(**inputs, **gen_args)
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract answer and reasoning
        try:
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else response.strip()
            choice = extract_choice(answer)
            
            reasoning_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Map choice to candidate
            if choice and choice in letters:
                idx = letters.index(choice)
                selected = candidates[idx] if idx < len(candidates) else None
                return selected, reasoning
            
            return None, reasoning
        except Exception as e:
            print(f"{RED}Error processing MCQ response: {e}{RESET}")
            return None, None
    
    def predict(self, image_path, question=None, num_sequences=20, top_k=5, data_type="bird"):
        """
        Complete two-step prediction pipeline
        
        Args:
            image_path: Path to input image
            question: Custom question (optional, uses default if None)
            num_sequences: Number of initial predictions to generate
            top_k: Number of top candidates to consider
            data_type: Type of object in image
            
        Returns:
            Dictionary with prediction results
        """
        # Default question for bird species recognition
        if question is None:
            question = f"""This is an image containing a {data_type}. Identify the species name of the {data_type}.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
The output answer format should be as follows:
<think> ... </think> <answer> species name </answer>
Please strictly follow the format."""
        
        # Step 1: Generate multiple predictions
        predictions = self.get_pass_at_k(image_path, question, num_sequences=num_sequences)
        
        if not predictions:
            print(f"{RED}No valid predictions generated!{RESET}")
            return {"error": "No predictions generated"}
        
        print(f"{GREEN}Generated {len(predictions)} unique predictions{RESET}")
        
        # Get top-k candidates
        candidates = self.get_top_k_candidates(predictions, k=top_k)
        print(f"{GREEN}Top {len(candidates)} candidates: {candidates}{RESET}")
        
        # Step 2: Refine with MCQ
        final_prediction, reasoning = self.refine_with_mcq(image_path, candidates, data_type=data_type)
        
        return {
            "all_predictions": predictions,
            "top_candidates": candidates,
            "final_prediction": final_prediction,
            "reasoning": reasoning
        }


def main():
    """Example usage"""
    # Initialize model
    model = DiVEInference(
        model_name="raja-kumar/DiVE-k-QWEN2.5-7B-CUB",
        device="cuda"  # Use "cpu" if no GPU available
    )
    
    # Example inference
    image_path = "./sample.jpg"  # Replace with your image path
    
    # Run prediction
    result = model.predict(
        image_path=image_path,
        num_sequences=20,  # Generate 20 predictions
        top_k=5,  # Consider top 5 candidates
        data_type="bird"
    )
    
    # Print results
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}RESULTS:{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")
    # print(f"\n{YELLOW}All predictions with frequencies:{RESET}")
    # for pred, freq in sorted(result['all_predictions'].items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {pred}: {freq}")
    
    print(f"\n{YELLOW}Top candidates:{RESET}")
    for i, cand in enumerate(result['top_candidates'], 1):
        print(f"  {i}. {cand}")
    
    print(f"\n{GREEN}Final prediction: {result['final_prediction']}{RESET}")
    
    if result.get('reasoning'):
        print(f"\n{YELLOW}Reasoning:{RESET}")
        print(f"  {result['reasoning']}")  # Print first 200 chars


if __name__ == "__main__":
    main()