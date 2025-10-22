# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess celeba dataset to parquet format for VERL training.
This script converts celeba classification data to the format expected by VERL.
"""

import argparse
from code import interact
import os
import json
import random
from typing import List, Dict, Any
# NEW: Added defaultdict for easier categorization
from collections import defaultdict

import datasets
from verl.utils.hdfs_io import copy, makedirs


def _read_image_bytes(image_path: str) -> bytes:
    """Read an image file and return raw bytes.

    Returns empty bytes (b"") on failure.
    """
    try:
        if not image_path or not os.path.isfile(image_path):
            return b""
        with open(image_path, "rb") as f:
            data = f.read()
        return data
    except Exception:
        return b""


def create_celeba_dataset_from_json(
    json_file: str,
    image_dir: str,
    split_mapping: Dict[int, str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create celeba dataset from JSON file.
    
    Args:
        json_file: Path to JSON file containing dataset information
        image_dir: Directory containing image files
        split_mapping: Mapping from split numbers to split names
    
    Returns:
        Dictionary with train, validation, and test datasets
    """
    if split_mapping is None:
        split_mapping = {0: "train", 1: "validation", 2: "test"}
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Define class names
    class_names = ["dark", "blond"]
    
    system_prompt = """Your goal is to accurately classify the hair color of the person in the provided image into one of two categories: "blond" or "dark" through a two-step process.

Category Definitions: "blond" is for blond hair; "dark" includes Black_Hair, Brown_Hair, and Gray_Hair, any hair color that is not "blond" should be classified as "dark".

The task will be completed in two distinct steps. You must strictly follow the instructions and adhere to the specific output format requested for each step.
"""

    user_prompt_celeba ="""* Step 1 Format: <box>[x1,y1,x2,y2]</box>

Step 1: Analyze the following image <image>. Your task is to generate a tight bounding box that covers only the person's hair. Provide the coordinates in the format <box>[x1,y1,x2,y2]</box>."""

    user_prompt_bg="""* Step 1 Format: <box>[x1,y1,x2,y2]</box>

Step 1: Analyze the following image <image>. Your task is to generate a tight bounding box that covers only the person's face. Provide the coordinates in the format <box>[x1,y1,x2,y2]</box>."""



    # Initialize datasets
    datasets_dict = {split_name: [] for split_name in split_mapping.values()}
    
    for item in data:
        # Extract information from JSON item
        image_path = item.get("img_filename", "")
        y = item.get("y", 0)  # 0 for landceleba, 1 for celeba
        split = item.get("split", 0)
        place=item.get("place",0)
        idx=item.get("img_id",0)
        # Map split number to split name
        split_name = split_mapping.get(split, "train")
        
        # Get class name
        class_name = class_names[y]
        
        # Create full image path
        full_image_path = os.path.join(image_dir, image_path) if image_path else ""
        # Prepare images field with raw bytes and relative path
        if full_image_path:
            image_entry = {
                "bytes": _read_image_bytes(full_image_path),
                "path": image_path,
            }
            images_field = [image_entry]
        else:
            images_field = []
        # print(full_image_path)
        # Create sample
        sample_celeba = {
            "data_source": "celeba",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": user_prompt_celeba
                }
            ],
            "images": images_field,
            "ability": "image_classification",
            "reward_model": {
                "style": "rule", 
                "ground_truth": class_name
            },
            "extra_info": {
                "idx":idx,
                "split": split,
                "class_name": class_name,
                "image_path": full_image_path,
                "interaction_kwargs":
                    {
                        "name":"celeba",
                        "img_path":full_image_path,
                        "ground_truth":class_name
                    },
                "is_negative":False,
                "place":place,
                "y":y
            },
        }
        sample_bg = {
            "data_source": "celeba",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": user_prompt_bg
                }
            ],
            "images": images_field,
            "ability": "image_classification",
            "reward_model": {
                "style": "rule", 
                "ground_truth": class_name
            },
            "extra_info": {
                "idx":idx,
                "split": split,
                "class_name": class_name,
                "image_path": full_image_path,
                "interaction_kwargs":
                    {
                        "name":"celeba",
                        "img_path":full_image_path,
                        "ground_truth":class_name
                    },
                "is_negative":True,
                "place":place,
                "y":y
            },
        }
        
        datasets_dict[split_name].append(sample_celeba)
        datasets_dict[split_name].append(sample_bg)
    return datasets_dict


# NEW: Function to perform stratified sampling on the training set
def stratified_sample_dataset(
    dataset: List[Dict[str, Any]], 
    ratio: int, 
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Performs stratified sampling based on 'y' and 'place' attributes.

    Args:
        dataset: The list of data samples to sample from.
        ratio: The inverse of the sampling fraction (e.g., 4 means 1/4).
        seed: Random seed for reproducibility.

    Returns:
        The new, sampled list of data samples.
    """
    if not isinstance(ratio, int) or ratio <= 0:
        raise ValueError("Sampling ratio must be a positive integer.")
    
    random.seed(seed)
    
    # Categorize data into the four groups
    categorized_data = defaultdict(list)
    for item in dataset:
        y = item["extra_info"]["y"]
        place = item["extra_info"]["place"]
        category = (y, place)
        categorized_data[category].append(item)
    
    # Print the original distribution
    print("\n--- Original Training Set Distribution ---")
    total_original = 0
    # Note: Corrected the user's prompt (y=1, place=1 was repeated) to cover all 4 possibilities.
    # The categories are (y=0, place=0), (y=0, place=1), (y=1, place=0), (y=1, place=1)
    for category, items in sorted(categorized_data.items()):
        count = len(items)
        print(f"Category (y={category[0]}, place={category[1]}): {count} samples")
        total_original += count
    print(f"Total original training samples: {total_original}\n")

    # Sample from each category and build the new dataset
    sampled_dataset = []
    for category, items in categorized_data.items():
        # Calculate how many samples to take from this category
        num_to_sample = len(items) // ratio
        # Randomly select samples and add them to the final list
        sampled_items = random.sample(items, k=num_to_sample)
        sampled_dataset.extend(sampled_items)
    
    # Print the new, sampled distribution for verification
    print(f"--- Sampled Training Set Distribution (1/{ratio} ratio) ---")
    new_categorized_data = defaultdict(list)
    for item in sampled_dataset:
        y = item["extra_info"]["y"]
        place = item["extra_info"]["place"]
        category = (y, place)
        new_categorized_data[category].append(item)
        
    total_sampled = 0
    for category, items in sorted(new_categorized_data.items()):
        count = len(items)
        print(f"Category (y={category[0]}, place={category[1]}): {count} samples")
        total_sampled += count
    print(f"Total sampled training samples: {total_sampled}\n")

    return sampled_dataset


def shuffle_dataset(dataset: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly shuffle the dataset.
    
    Args:
        dataset: List of data samples
        seed: Random seed for reproducibility
    
    Returns:
        Shuffled dataset
    """
    random.seed(seed)
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    return shuffled_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess celeba dataset for VERL")
    parser.add_argument("--json_file", default="/data/wangnn/repos/verl/debiasing/data_celeba.json",
                        help="Path to JSON file containing dataset information (e.g., data_celeba.json)")
    parser.add_argument("--image_dir", default="/data/wangnn/datasets/celeba",required=False,
                        help="Directory containing image files")
    parser.add_argument("--output_dir", default="/data/wangnn/repos/verl/debiasing/data",
                        help="Output directory for processed data")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for output")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle the dataset (default: True)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    # NEW: Add argument for sampling ratio
    parser.add_argument("--sampling_ratio", type=int, default=None,
                        help="Ratio for stratified sampling of the training set. E.g., 4 means 1/4 of the data will be sampled.")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process celeba dataset from JSON file
    print(f"Processing celeba dataset from {args.json_file}...")
    datasets_dict = create_celeba_dataset_from_json(
        args.json_file, 
        args.image_dir
    )
    
    # NEW: Perform stratified sampling on the training set if requested
    if args.sampling_ratio:
        datasets_dict["train"] = stratified_sample_dataset(
            dataset=datasets_dict["train"],
            ratio=args.sampling_ratio,
            seed=args.seed
        )

    # Shuffle datasets if requested
    if args.shuffle:
        print("Shuffling datasets...")
        for split_name in datasets_dict:
            datasets_dict[split_name] = shuffle_dataset(
                datasets_dict[split_name], 
                args.seed
            )
    
    # Convert to datasets.Dataset
    train_ds = datasets.Dataset.from_list(datasets_dict["train"])
    val_ds = datasets.Dataset.from_list(datasets_dict["validation"])
    test_ds = datasets.Dataset.from_list(datasets_dict["test"])
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Save to parquet
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    test_ds.to_parquet(test_path)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved validation data to {val_path}")
    print(f"Saved test data to {test_path}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        # makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()