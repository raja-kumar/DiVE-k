from datasets import interleave_datasets, concatenate_datasets
from datasets import DatasetDict

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def prepare_datasets(script_args, use_hard_examples=False, normal_to_hard_ratio=2):
    """
    Prepares the dataset for training based on the configuration.

    Args:
        script_args: The script arguments containing dataset paths and splits.
        use_hard_examples (bool): Whether to include hard examples in the training dataset.
        normal_to_hard_ratio (int): The ratio of normal to hard examples in the combined dataset.

    Returns:
        A combined dataset if `use_hard_examples` is True, otherwise the normal dataset.
    """
    # Load the normal dataset
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)

    # Format normal dataset
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("DEBUG: Normal dataset contains images.")
        dataset = dataset.map(make_conversation_image)
    else:
        print("DEBUG: Normal dataset does not contain images.")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    print(f"DEBUG: Normal dataset size: {len(dataset[script_args.dataset_train_split])}")

    if not use_hard_examples or not script_args.hard_dataset_name:
        # Return only the normal dataset
        print("DEBUG: Using only the normal dataset.")
        return dataset[script_args.dataset_train_split]

    # Load the hard dataset
    hard_dataset = DatasetDict.load_from_disk(script_args.hard_dataset_name)

    # Format hard dataset
    if "image" in hard_dataset[script_args.dataset_train_split].features:
        print("DEBUG: Hard dataset contains images.")
        hard_dataset = hard_dataset.map(make_conversation_image)
    else:
        print("DEBUG: Hard dataset does not contain images.")
        hard_dataset = hard_dataset.map(make_conversation)
        hard_dataset = hard_dataset.remove_columns("messages")

    print(f"DEBUG: Hard dataset size: {len(hard_dataset[script_args.dataset_train_split])}")

    

    # Repeat hard dataset to match the size of the normal dataset
    normal_size = len(dataset[script_args.dataset_train_split])
    hard_size = len(hard_dataset[script_args.dataset_train_split])

    # if (hard_size > 100):
    #     print("DEBUG: Hard dataset is large, using only the first 100 examples.")
    #     hard_dataset = hard_dataset[script_args.dataset_train_split].select(range(100))
    #     hard_size = len(hard_dataset)
    
    repeat_factor = (normal_size + hard_size - 1) // hard_size  # Calculate repeat factor
    repeated_hard_dataset = concatenate_datasets(
        [hard_dataset[script_args.dataset_train_split]] * repeat_factor
    )

    # Truncate repeated hard dataset to match the exact size
    repeated_hard_dataset = repeated_hard_dataset.select(range(normal_size))

    print(f"DEBUG: Repeated hard dataset size after truncation: {len(repeated_hard_dataset)}")

    # Interleave datasets with the specified ratio
    combined_dataset = interleave_datasets(
        [dataset[script_args.dataset_train_split], repeated_hard_dataset],
        probabilities=[normal_to_hard_ratio / (normal_to_hard_ratio + 1), 1 / (normal_to_hard_ratio + 1)],
        seed=42
    )

    print(f"DEBUG: Combined dataset size: {len(combined_dataset)}")

    if len(combined_dataset) > 2400:
        print("DEBUG: Combined dataset is larger than 2400, truncating to 2400 examples.")
        combined_dataset = combined_dataset.select(range(2400))
    
    print(f"DEBUG: Final Combined dataset size: {len(combined_dataset)}")
    
    return combined_dataset