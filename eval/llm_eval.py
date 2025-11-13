import json
from openai import OpenAI
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import sys
import re
import argparse

class AccuracyEvaluator:
    def __init__(self, api_key: str, one_answer: bool = True, one_step:bool = False):
        # self.client = OpenAI(api_key=api_key)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.one_answer = one_answer
        self.one_step = one_step
        self.model = "google/gemini-2.5-flash-lite-preview-06-17" # or "google/gemini-2.5-flash-lite-preview-06-17" for Gemini
        
    def _check_match(self, groundtruth: str, predictions: str):
        """Use LLM to determine if any prediction matches groundtruth."""

        if (len(predictions) == 0):
            print("Empty prediction")
            return False, "Empty prediction"
        
        if self.one_answer:
            prompt = f"""You are evaluating fine-grained image classification results.
    Given:
    - Groundtruth category: "{groundtruth}"
    - LLM prediction: {predictions}

    Check if the groundtruth matches the prediction. The strings need not match exactly but they must refer to the same specific fine-grained category, not just broad class.
    Respond with:
    1. "True" or "False" if groundtruth matches the prediction in <answer></answer> tag. i.e <answer>answer here (True/False)</answer>
    2. Brief explanation in <explanation></explanation> tag. i.e <explanation>Explanation here</explanation>
    """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            
            try:
                result = response.choices[0].message.content
                top1_match = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[0].strip()
            except Exception as e:
                print(f"Error in response: {e}")
                return False, "Error in response"
            if (top1_match.lower() == "true"):
                top1_match = True
            else:
                top1_match = False
            try:
                explanation = re.findall(r'<explanation>(.*?)</explanation>', result, re.DOTALL)[0].strip()
            except IndexError:
                explanation = "No explanation provided"
                print(result)
            
            return top1_match, explanation
        else:
            prompt = f"""You are evaluating fine-grained image classification results.
    Given:
    - Groundtruth category: "{groundtruth}"
    - Predicted categories: {predictions}

    Your goal is to find the top-k correctness, where k is the number of predictions in Predicted categories. The strings need not match exactly but they must refer to the same specific category (not just broad class). 
    if the prediction is more specific (or fine-grained) than the groundtruth, it is considered correct.
    Use your knowledge about scientific names and common names of plants to determine if the groundtruth matches any of the predictions.

    The output answer format should be as follows:
    <answer>answer</answer>
    

    answer is "True" or "False" if groundtruth matches any of the predictions. 

    Please strictly follow the format.
    """
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content



            try:
                top5_match = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[0].strip()

                top5_match = "true" in top5_match.lower() and "false" not in top5_match.lower()

                return False, top5_match, ""
            
            except IndexError:
                print("Error parsing response:")
                return False, False, "Error parsing response"

    def evaluate_predictions(self, data: Dict, step3_data: Dict = None) -> Dict:
        """Evaluate predictions and compute top-1 and top-5 accuracy metrics."""
        total = len(data)
        correct_top1 = 0
        correct_top5 = 0
        correct_consistency = 0
        results = {}

        for image_id, item in tqdm(data.items()):
            groundtruth = item["groundtruth"]

            if self.one_step:
                predictions = item["answer"]
                top_prediction = predictions
            else:
                # Check if step3_data is provided and contains the current image_id
                if step3_data and image_id in step3_data:
                    step3_item = step3_data[image_id]
                    if step3_item["consistent_prediction"].lower() == "true":
                        predictions = step3_item["original_prediction"]
                    else:
                        predictions = step3_item["corrected_prediction"]
                else:
                    predictions = item["prediction"]
                step1_output = item["step1_output"]
                top_prediction = sorted(step1_output.items(), key=lambda x: x[1], reverse=True)[0][0] if step1_output else None

            if self.one_answer:
                top1_match, top1_explanation = self._check_match(groundtruth, predictions)
                consistency_match, consistency_explanation = self._check_match(groundtruth, top_prediction)
                top5_match = top1_match
            else:
                top1_match, top5_match, explanation = self._check_match(groundtruth, predictions)

            results[image_id] = {
                "groundtruth": groundtruth,
                "predictions": predictions,
                "consistency_prediction": top_prediction,
                "top1_correct": top1_match,
                "top5_correct": top5_match,
                "consistency_correct": consistency_match,
                "top1_explanation": top1_explanation,
                "consistency_explanation": consistency_explanation,
            }

            if top1_match:
                correct_top1 += 1
            if top5_match:
                correct_top5 += 1
            if consistency_match:
                correct_consistency += 1
            
        return {
            "top1_accuracy": correct_top1 / total,
            "top5_accuracy": correct_top5 / total,
            "consistency_accuracy": correct_consistency / total,
            "total_samples": total,
            "correct_top1": correct_top1,
            "correct_top5": correct_top5,
            "correct_consistency": correct_consistency,
            "detailed_results": results
        }

def main(output_file: str = "predictions.json", one_answer: bool = True, one_step: bool = False, step3_file: str = None):
    # Load API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")

    # Initialize evaluator
    evaluator = AccuracyEvaluator(api_key, one_answer, one_step)

    # Load predictions
    with open(output_file, "r") as f:
        predictions = json.load(f)

    # Load step3_file if provided
    step3_data = None
    if step3_file:
        with open(step3_file, "r") as f:
            step3_data = json.load(f)

    # Evaluate predictions
    results = evaluator.evaluate_predictions(predictions, step3_data)

    # Print results
    print(f"\nTop-1 Accuracy: {results['top1_accuracy']:.2%}")
    # print(f"Top-5 Accuracy: {results['top5_accuracy']:.2%}")
    print(f"Consistency Accuracy: {results['consistency_accuracy']:.2%}")
    print(f"Correct predictions (Top-1): {results['correct_top1']}/{results['total_samples']}")
    # print(f"Correct predictions (Top-5): {results['correct_top5']}/{results['total_samples']}")
    print(f"Correct predictions (Consistency): {results['correct_consistency']}/{results['total_samples']}")

    # Save detailed results
    if step3_file:
        file_name = output_file.replace(".json", f"_evaluation_with_step3.json")
    else:
        file_name = output_file.replace(".json", "_evaluation.json")
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions for fine-grained classification.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to predictions JSON file.")
    parser.add_argument("--one_answer", type=str, default="true", help="Whether to use one answer mode (true/false).")
    parser.add_argument("--one_step", type=str, required=True, help="True for one step inference output")
    parser.add_argument("--step3_file", type=str, help="Path to step 3 JSON file.")
    args = parser.parse_args()

    one_answer = args.one_answer.lower() == "true"
    one_step = args.one_step.lower() == "true"
    print(f"Evaluating with one_answer={one_answer}, one_step={one_step}, step3_file={args.step3_file}")
    main(args.output_file, one_answer, one_step, args.step3_file)