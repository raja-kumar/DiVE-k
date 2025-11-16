PROMPTS = {
    "oxford_flowers": {
        "instruction": "output the most likely species name in the image.",
        "answer_format": "species name",
        "data_name": "flower",
    },
    "oxford-iiit-pet": {
        "instruction": "output the most likely species name in the image.",
        "answer_format": "species name",
        "data_name": "pet",
    },
    "stanford_cars": {
        "instruction": "output the most likely make and model of the car in the image.",
        "answer_format": "make model",
        "data_name": "car",
    },
    "fgvc_aircraft": {
        "instruction": "output the most likely make and model of the aircraft in the image.",
        "answer_format": "make model",
        "data_name": "aircraft",
    },
    "CUB_200_2011": {
        "instruction": "output the most likely species name in the image.",
        "answer_format": "species name",
        "data_name": "bird",
    },
    'gqa': {
        "instruction": "output the most likely answer to the question based on the image.",
        "answer_format": "final answer",
        "data_name": "gqa",
    }
}

prompts = {
    "oxford_flowers": " This is an image containing a flower or flower plant. Please identify the species of the flower based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>species name</answer>\nPlease strictly follow the format.",
    "oxford-iiit-pets": " This is an image containing a pet. Please identify the species of the pet based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>species name</answer>\nPlease strictly follow the format.",
    "stanford_cars": " This is an image containing a car. Please identify the model of the car which should include year, make and model based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>make model</answer>\nPlease strictly follow the format.",
    "fgvc_aircraft": " This is an image containing an aircraft. Please identify the model of the aircraft based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>make model</answer>\nPlease strictly follow the format.",
    "CUB_200_2011": "This is an image containing a bird. Please identify the species of the bird based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>species name</answer>\nPlease strictly follow the format.",
}