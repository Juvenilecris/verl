import json
from typing import Optional, Dict, Any
import re
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Optional
import torch

def find_last_occurrence(input_tensor: torch.Tensor, search_values: list) -> int:
    """
    在一个一维 Tensor 中，查找一个列表里多个数值中任意一个最后出现的位置索引。

    Args:
        input_tensor (torch.Tensor): 需要在其中搜索的一维张量。
        search_values (list): 包含一个或多个要搜索的数值的列表。

    Returns:
        int: 列表中任意数值在 Tensor 中最后一次出现的索引。
             如果 Tensor 中不存在列表中的任何数值，则返回 -1。
    """
    mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    for value in search_values:
        mask = mask | (input_tensor == value)
    
    indices = torch.where(mask)[0]
    
    if indices.numel() > 0:
        return indices[-1].item()
    else:
        return -1
    

def computer_bbox_score(solution_str,ground_truth)->float:
    box_pattern = re.compile(r"<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>")
    match = box_pattern.search(solution_str)
    return 1.0 if match else 0.0
    
def extract_answer(solution: str) -> str:
    """
    Extracts the final answer from a string containing multiple <answer>...</answer> tags.

    Args:
        solution: The input string containing the answer within <answer>...</answer> tags.

    Returns:
        The text content of the last <answer>...</answer> tag, or an empty string
        if no such tag is found.
    """
    matches = re.findall(r'<answer>(.*?)</answer>', solution, re.DOTALL)
    if matches:
        return matches[-1]
    return solution


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    # Use torch.no_grad() to prevent building the computation graph
    with torch.no_grad():
        if "log_probs_token_ids" in extra_info.keys():
            log_probs_token_ids = extra_info["log_probs_token_ids"].cpu()
            probs_token_ids = torch.exp(log_probs_token_ids)
            output_tokens_ids = extra_info["output_tokens_ids"].cpu()
            index = find_last_occurrence(output_tokens_ids, [2024,22624])
            new_log_probs_token_ids = torch.stack([
                log_probs_token_ids[0] + torch.cat([
                    log_probs_token_ids[1, 1:],
                    torch.tensor([0.0], dtype=log_probs_token_ids.dtype, device=log_probs_token_ids.device)
                ]),
                log_probs_token_ids[2]
            ])
            probs_token_ids = torch.exp(new_log_probs_token_ids)
            output_tokens_ids = extra_info["output_tokens_ids"].cpu()
            if index == -1:
                return -1.0
                
        
        bbox_rewards = sum(extra_info["bbox_rewards"]) if "bbox_rewards" in extra_info.keys() else 0.0
        if bbox_rewards == -1:
            return -1.0
            
        brid_dict = {"dark": 1, "blond": 0}
        ground_truth = ground_truth.strip().lower() if isinstance(ground_truth, str) else ""
        
        try:
            solution = solution_str.split('assistant')[-1]
            vlm_answer = extract_answer(solution)
            vlm_answer = vlm_answer.strip().lower() if isinstance(vlm_answer, str) else None
            assert vlm_answer in ['blond', 'dark']
            if extra_info['is_negative']:
                prob_in_gt = 1 - probs_token_ids[brid_dict[ground_truth]][index]
            else:    
                prob_in_gt = probs_token_ids[brid_dict[ground_truth]][index]
            return prob_in_gt.item() + bbox_rewards
                
            
        except (Exception, AssertionError) as e:
            print(e)
            return -1.0 + bbox_rewards