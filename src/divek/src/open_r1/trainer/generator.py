## get model, input
## use the model and input to sample k outputs
## use sampled output to genrate mcq options
from open_r1.trainer.prompts import PROMPTS, prompts
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
import re
import json
import random
from open_r1.trainer.utils import (
    clean_string, clean_topk, post_process_passk, process_generated_output, sample_random_options, build_mcq_prompt,
    format_options, find_mcq_answer, pred_class_to_idx, create_prompt
)
import torch
import copy

class TopKGenerator:
    def __init__(self, class_to_idx, data_name, processing_class=None, category_names_path=None):
        self.class_to_idx = {clean_string(k): v for k, v in class_to_idx.items()}
        with open(category_names_path, 'r') as f:
            self.categories = [clean_string(cat) for cat in f.read().splitlines()]
        self.data_name = data_name
        self.processing_class = processing_class
        self.input_prompt = create_prompt(self.categories, data_name)
        self.promt_inputs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.input_prompt},
                    ],
                },
            ]

    def generate_mcq(self, curr_pred, gt):
        gt_cat_name = clean_string(re.search("<answer>(.*?)</answer>", gt).group(1))

        try:
            gpt_preds = list(post_process_passk(curr_pred).keys())
        except Exception:
            gpt_preds = sample_random_options(self.categories, gt_cat_name, 2)
        
        gpt_preds = clean_topk(gpt_preds, self.class_to_idx)
        gt_label = pred_class_to_idx([gt_cat_name], self.class_to_idx)[0]
        gpt_labels = pred_class_to_idx(gpt_preds, self.class_to_idx)

        # Add random options if too few
        if len(gpt_preds) <= 1:
            random_options = sample_random_options(self.categories, gt_cat_name, 2)
            gpt_preds += random_options

        # Ensure gt_cat_name is in options
        if gt_label == -1 or (gt_label not in gpt_labels):
            if len(gpt_preds) < 5:
                gpt_preds.append(gt_cat_name)
            else:
                gpt_preds[-1] = gt_cat_name
        
        random.shuffle(gpt_preds)
        options_str = format_options(gpt_preds)
        answer_letter = find_mcq_answer(gpt_preds, gt_cat_name)

        data_name = PROMPTS[self.data_name]["data_name"]
        prompt = build_mcq_prompt(data_name, options_str)

        return prompt, f"<answer>{answer_letter}</answer>"


    def get_topk(self, model, inputs):        

        generation_args = {
                "max_new_tokens": 1024,
                "temperature": 1.0,
                "top_p": 0.95,
                "do_sample": True,
                "num_return_sequences": 10,
                "repetition_penalty": 1.1,
                "use_cache": True,
            }
        
        solutions = []
        prompts = []
        for input_ in inputs:
            input_["prompt"] = self.promt_inputs

            image = [input_["image"]]
            gt = input_["solution"]


            prompts_text = [maybe_apply_chat_template(input_, self.processing_class)["prompt"]]

            inputs = self.processing_class(
                text=prompts_text,
                images=image,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )

            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generation_args)
            
            input_id_length = inputs.input_ids.shape[1]
            num_sequences = generation_args["num_return_sequences"]
            answer_format = PROMPTS[self.data_name]["answer_format"]
            curr_pred = process_generated_output(
                generated_ids, input_id_length, num_sequences, self.processing_class, answer_format
            )

            print(curr_pred)
            prompt, answer = self.generate_mcq(curr_pred, gt)

            model_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            input_["prompt"] = model_prompt
            input_["solution"] = answer
            
            solutions.append(answer)
            prompts.append(model_prompt)
        
        return prompts, solutions