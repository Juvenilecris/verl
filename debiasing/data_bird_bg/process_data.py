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
Preprocess waterbird dataset to parquet format for VERL training.
This script converts waterbird classification data to the format expected by VERL,
with custom ordering and augmentation for train/validation sets.
"""

import argparse
import os
import json
import random
from typing import List, Dict, Any

import datasets
from verl.utils.hdfs_io import copy, makedirs # 注意：如果您不使用HDFS，可以忽略此行


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


def create_waterbird_dataset_from_json(
    json_file: str,
    image_dir: str,
    split_mapping: Dict[int, str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create waterbird dataset from JSON file.
    
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
    class_names = ["landbird", "waterbird"]
    
    # Define prompts based on prompt.xml
    system_prompt = """Your goal is to accurately classify a bird in an image as either a **Waterbird** or a **Landbird** through a two-step process.

Category Definitions: Waterbirds include, but are not limited to, the following species: albatross, auklet, cormorant, frigatebird, fulmar, gull, jaeger, kittiwake, pelican, puffin, tern, gadwall, grebe, mallard, merganser, guillemot, or Pacific loon; Landbirds are all other bird species that are not categorized as Waterbirds.

The task will be completed in two distinct steps. You must strictly follow the instructions and adhere to the specific output format requested for each step.
"""
    
    user_prompt_bird = """* Step 1 Format: <box>[x1,y1,x2,y2]</box>

Step 1: Analyze the following image <image>. Your task is to generate a tight bounding box that covers only the bird with minimal background. Provide the coordinates in the format <box>[x1,y1,x2,y2]</box>."""
    
    user_prompt_bg="""* Step 1 Format: <box>[x1,y1,x2,y2]</box>

Step 1: Analyze the following image <image>. Your task is to generate a bounding box that best represents the image background while excluding birds, ensuring the box is as large as possible and no part of any bird falls within this bounding box. Provide the coordinates in the format <box>[x1,y1,x2,y2]</box>."""
    
    # Initialize datasets
    datasets_dict = {split_name: [] for split_name in split_mapping.values()}
    
    for item in data:
        # Extract information from JSON item
        image_path = item.get("img_filename", "")
        y = item.get("y", 0)  # 0 for landbird, 1 for waterbird
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
        
        # Create sample_bird (is_negative=False)
        sample_bird = {
            "data_source": "waterbird",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_bird}
            ],
            "images": images_field,
            "ability": "image_classification",
            "reward_model": {"style": "rule", "ground_truth": class_name},
            "extra_info": {
                "idx":idx, "split": split, "class_name": class_name,
                "image_path": full_image_path,
                "interaction_kwargs": {
                    "name":"waterbird", "img_path":full_image_path, "ground_truth":class_name
                },
                "is_negative":False, "place":place, "y":y
            },
        }
        
        # Create sample_bg (is_negative=True)
        sample_bg = {
            "data_source": "waterbird",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_bg}
            ],
            "images": images_field,
            "ability": "image_classification",
            "reward_model": {"style": "rule", "ground_truth": class_name},
            "extra_info": {
                "idx":idx, "split": split, "class_name": class_name,
                "image_path": full_image_path,
                "interaction_kwargs": {
                    "name":"waterbird", "img_path":full_image_path, "ground_truth":class_name
                },
                "is_negative":True, "place":place, "y":y
            },
        }
        
        datasets_dict[split_name].append(sample_bird)
        datasets_dict[split_name].append(sample_bg)
    return datasets_dict


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
    parser = argparse.ArgumentParser(description="Preprocess waterbird dataset for VERL")
    parser.add_argument("--json_file", required=True,
                        help="Path to JSON file containing dataset information (e.g., data_waterbird.json)")
    parser.add_argument("--image_dir", default="/data/wangnn/datasets/waterbrids/waterbird", required=False,
                        help="Directory containing image files")
    parser.add_argument("--output_dir", default="/data/wangnn/repos/verl/debiasing/data",
                        help="Output directory for processed data")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for output")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle the dataset (default: True)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process waterbird dataset from JSON file
    print(f"Processing waterbird dataset from {args.json_file}...")
    datasets_dict = create_waterbird_dataset_from_json(
        args.json_file, 
        args.image_dir
    )
    
    # ==============================================================================
    # START: 自定义排序与扩充逻辑
    # ==============================================================================
    print("\nApplying custom ordering and augmentation for train/validation sets...")

    # 1. 合并 train 和 validation 数据
    combined_train_val_list = datasets_dict["train"] + datasets_dict["validation"]
    print(f"Combined train and validation sets. Total samples: {len(combined_train_val_list)}")

    # 2. 根据 'is_negative' 标志分离数据
    data_false = [sample for sample in combined_train_val_list if not sample['extra_info']['is_negative']]
    data_true = [sample for sample in combined_train_val_list if sample['extra_info']['is_negative']]
    print(f"Separated data: {len(data_false)} samples with is_negative=False, {len(data_true)} samples with is_negative=True.")

    # 3. 构建三个部分
    # Part 1: is_negative=False 的数据重复4次，然后内部随机打乱
    part1 = data_false * 4
    if args.shuffle:
        print("Shuffling Part 1 internally...")
        part1 = shuffle_dataset(part1, seed=args.seed)
    print(f"Part 1 (is_negative=False, repeated 4 times, internally shuffled) contains {len(part1)} samples.")

    # Part 2: is_negative=True 的数据重复4次，然后内部随机打乱
    part2 = data_true * 4
    if args.shuffle:
        print("Shuffling Part 2 internally...")
        # 使用不同的种子确保与part1的随机性独立
        part2 = shuffle_dataset(part2, seed=args.seed + 1)
    print(f"Part 2 (is_negative=True, repeated 4 times, internally shuffled) contains {len(part2)} samples.")
    
    # Part 3: 原始混合数据随机打乱后重复12次
    # (这里的 shuffle 逻辑保持不变)
    shuffled_combined_data = shuffle_dataset(combined_train_val_list, seed=args.seed) if args.shuffle else combined_train_val_list
    part3 = shuffled_combined_data * 6
    print(f"Part 3 (shuffled mix, repeated 12 times) contains {len(part3)} samples.")

    # 4. 拼接最终的训练数据集
    final_train_list = part1 + part2 + part3
    print(f"Final combined training dataset created with {len(final_train_list)} total samples.")

    # 5. 转换为 datasets.Dataset 并保存
    final_train_ds = datasets.Dataset.from_list(final_train_list)
    final_train_path = os.path.join(args.output_dir, "final_train.parquet")
    final_train_ds.to_parquet(final_train_path)
    print(f"\nSaved final combined and augmented train data to {final_train_path}")

    # ==============================================================================
    # END: 逻辑结束
    # ==============================================================================
    
    # 处理并保存独立的测试集 (逻辑保持不变)
    print("\nProcessing and saving the test set separately...")
    test_list = datasets_dict["test"]
    if args.shuffle:
        print("Shuffling test set...")
        test_list = shuffle_dataset(test_list, args.seed)
        
    test_ds = datasets.Dataset.from_list(test_list)
    print(f"Test samples: {len(test_ds)}")
    
    test_path = os.path.join(args.output_dir, "test.parquet")
    test_ds.to_parquet(test_path)
    print(f"Saved test data to {test_path}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        # makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()