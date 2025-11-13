import json
import os
import re
import argparse
import sys
from typing import Dict, List, Tuple
from openai import OpenAI
from tqdm import tqdm

class AccuracyEvaluator:
    """
    Evaluates fine-grained classification predictions using an LLM to compare
    predicted labels against ground truth labels.
    """

    def __init__(self, api_key: str, one_step: bool, eval_topk: bool):
        """
        Initializes the evaluator.

        Args:
            api_key (str): The API key for the LLM service (e.g., OpenRouter).
            one_step (bool): Flag indicating if predictions are from a one-step process.
            eval_topk (bool): Flag to enable exclusive top-k evaluation.
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "google/gemini-2.5-flash-lite-preview-06-17"
        self.one_step = one_step
        self.eval_topk = eval_topk

    def _check_top1_match(self, groundtruth: str, prediction: str) -> Tuple[bool, str]:
        """Uses an LLM to determine if a single prediction matches the ground truth."""
        if not prediction:
            return False, "No prediction provided."

        prompt = f"""
        You are an expert evaluator for fine-grained image classification.
        Your task is to determine if the predicted category matches the groundtruth category.

        - A match is TRUE if the prediction refers to the same specific category as the groundtruth.
        - An exact string match is not required. Synonyms or scientific/common name equivalents are matches.

        <groundtruth>{groundtruth}</groundtruth>
        <prediction>{prediction}</prediction>

        Provide your response in the following format:
        <answer>[True or False]</answer>
        <explanation>[A brief explanation for your decision]</explanation>
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=200
            )
            result = response.choices[0].message.content
            match_str = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL).group(1).strip()
            explanation = re.search(r'<explanation>(.*?)</explanation>', result, re.DOTALL).group(1).strip()
            is_match = "true" in match_str.lower()
            return is_match, explanation
        except (AttributeError, IndexError) as e:
            print(f"Error parsing LLM response for top-1: {e}\nResponse: {result}")
            return False, "Error in parsing LLM response."
        except Exception as e:
            print(f"An API error occurred: {e}")
            return False, "API call failed."

    def _check_topk_match(self, groundtruth: str, predictions: List[str]) -> Tuple[bool, str]:
        """Uses an LLM to determine if the ground truth is present in a list of predictions."""
        if not predictions:
            return False, "No predictions provided."

        predictions_str = "\n".join([f"- {p}" for p in predictions])
        prompt = f"""
        You are an expert evaluator for fine-grained image classification.
        Your task is to determine if the groundtruth category is present in the list of predicted categories.

        - A match is TRUE if any prediction refers to the same specific category as the groundtruth.
        - An exact string match is not required. Synonyms or scientific/common name equivalents are matches.

        <groundtruth>{groundtruth}</groundtruth>
        <predictions>
        {predictions_str}
        </predictions>

        Provide your response in the following format:
        <answer>[True or False]</answer>
        <explanation>[If True, state which prediction matched. If False, give a brief reason.]</explanation>
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=200
            )
            result = response.choices[0].message.content
            match_str = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL).group(1).strip()
            explanation = re.search(r'<explanation>(.*?)</explanation>', result, re.DOTALL).group(1).strip()
            is_match = "true" in match_str.lower()
            return is_match, explanation
        except (AttributeError, IndexError) as e:
            print(f"Error parsing LLM response for top-k: {e}\nResponse: {result}")
            return False, "Error in parsing LLM response."
        except Exception as e:
            print(f"An API error occurred: {e}")
            return False, "API call failed."

    def evaluate_predictions(self, data: Dict) -> Dict:
        """Evaluates predictions to compute accuracy metrics based on the configured mode."""
        total_samples = len(data)
        correct_top1 = 0
        correct_topk = 0
        correct_consistency = 0
        detailed_results = {}

        for image_id, item in tqdm(data.items(), desc="Evaluating predictions"):
            groundtruth = item["groundtruth"]
            
            # --- Exclusive Top-k Evaluation Mode ---
            if self.eval_topk:
                predictions = item["step1_output"]
                topk_match, topk_explanation = self._check_topk_match(groundtruth, predictions)
                if topk_match:
                    correct_topk += 1
                result_item = {
                    "groundtruth": groundtruth,
                    "predictions": predictions,
                    "topk_correct": topk_match,
                    "topk_explanation": topk_explanation,
                }
            # --- Default Evaluation Mode (Top-1 and Consistency) ---
            else:
                if self.one_step:
                    predictions = [item["answer"]]
                    consistency_pred = None
                else:
                    predictions = item["prediction"]
                    step1_output = item.get("step1_output", {})
                    consistency_pred = sorted(step1_output.items(), key=lambda x: x[1], reverse=True)[0][0] if step1_output else None

                top1_prediction = predictions[0] if predictions else None
                top1_match, top1_explanation = self._check_top1_match(groundtruth, top1_prediction)
                
                result_item = {
                    "groundtruth": groundtruth,
                    "predictions": predictions,
                    "top1_correct": top1_match,
                    "top1_explanation": top1_explanation,
                }
                if top1_match:
                    correct_top1 += 1

                if not self.one_step:
                    consistency_match, consistency_explanation = self._check_top1_match(groundtruth, consistency_pred)
                    if consistency_match:
                        correct_consistency += 1
                    result_item["consistency_prediction"] = consistency_pred
                    result_item["consistency_correct"] = consistency_match
                    result_item["consistency_explanation"] = consistency_explanation
            
            detailed_results[image_id] = result_item

        # --- Compile Final Report based on the mode ---
        final_report = {"total_samples": total_samples}
        if self.eval_topk:
            final_report["topk_accuracy"] = correct_topk / total_samples if total_samples > 0 else 0
            final_report["correct_topk"] = correct_topk
        else:
            final_report["top1_accuracy"] = correct_top1 / total_samples if total_samples > 0 else 0
            final_report["correct_top1"] = correct_top1
            if not self.one_step:
                final_report["consistency_accuracy"] = correct_consistency / total_samples if total_samples > 0 else 0
                final_report["correct_consistency"] = correct_consistency
        
        final_report["detailed_results"] = detailed_results
        return final_report

def main():
    """Main function to run the evaluation from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM predictions for fine-grained classification using an LLM evaluator."
    )
    parser.add_argument(
        "--output-file", type=str, required=True,
        help="Path to the JSON file containing predictions."
    )
    parser.add_argument(
        "--one-step", action=argparse.BooleanOptionalAction, default=False,
        help="Specify if the prediction file is from a one-step inference process."
    )
    parser.add_argument(
        "--eval-topk", action="store_true",
        help="Run evaluation in exclusive top-k mode. Cannot be used with --one-step."
    )
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.eval_topk and args.one_step:
        sys.exit("ERROR: --eval-topk cannot be used with --one-step, as top-k is not applicable to single-answer predictions.")

    # Load API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

    # Initialize evaluator
    evaluator = AccuracyEvaluator(
        api_key=api_key, 
        one_step=args.one_step, 
        eval_topk=args.eval_topk
    )

    # Load prediction data
    # print(f"Loading predictions from: {args.output_file}")
    with open(args.output_file, "r") as f:
        predictions_data = json.load(f)

    # Run the evaluation
    results = evaluator.evaluate_predictions(predictions_data)

    # --- Print Final Results based on the mode ---
    print(f"\n--- Evaluation Results for {args.output_file} ---")
    print(f"Total Samples: {results['total_samples']}")
    
    if args.eval_topk:
        print(f"Top-k Accuracy: {results['topk_accuracy']:.2%}")
        print(f"Correct (Top-k): {results['correct_topk']}/{results['total_samples']}")
        evaluation_file_path = args.output_file.replace(".json", "_passk_evaluation.json")
    else:
        evaluation_file_path = args.output_file.replace(".json", "_evaluation.json")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.2%}")
        print(f"Correct (Top-1): {results['correct_top1']}/{results['total_samples']}")
        if not args.one_step:
            print(f"Consistency Accuracy: {results['consistency_accuracy']:.2%}")
            print(f"Correct (Consistency): {results['correct_consistency']}/{results['total_samples']}")
            
    print("------------------------\n")

    # Save detailed evaluation results
    
    with open(evaluation_file_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed evaluation results saved to: {evaluation_file_path}")

if __name__ == "__main__":
    main()