from datasets import DatasetDict, Dataset

def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)

test_dataset_dict = load_dataset("/data2/raja/oxford_flowers/zero_shot/subsample_base_train_sample_dataset")

print(test_dataset_dict["train"])