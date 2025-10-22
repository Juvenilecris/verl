# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import re
from PIL import Image
from verl.utils.reward_score import gsm8k
from verl.interactions.qwenvl_utils import smart_resize,reverse_convert_to_original_format

from .base import BaseInteraction
import cv2
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class CelebAInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of gsm8k.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "img_path":kwargs.pop('img_path', None)
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        self._instance_dict[instance_id]["response"] = content
        img_path=self._instance_dict[instance_id]['img_path']
        content=content.replace(", ", ",")
        try:
            box_pattern = re.compile(r"<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>")
            match = box_pattern.search(content)
            if match:
                reward=1.0
                coords_str = match.groups()
                box_coords = [int(c) for c in coords_str] # -> [x1, y1, x2, y2]
                image = cv2.imread(img_path)
                orig_height = image.shape[0]
                orig_width = image.shape[1]
                new_height, new_width = smart_resize(orig_height, orig_width)
                box_coords=reverse_convert_to_original_format(box_coords,orig_height, orig_width, new_height, new_width)
                assert box_coords[0]+5<box_coords[2]
                assert box_coords[1]+5<box_coords[3]
                left, upper, right, lower = box_coords
                if left < 0 or upper < 0 or right > orig_width+5 or lower > orig_height+5:
                    raise ValueError(
                        f"Crop coordinates {box_coords} are outside the image dimensions ({orig_width}, {orig_height})."
                    )
                original_image = Image.open(img_path).convert("RGB")
                cropped_image = original_image.crop(box_coords)
                from verl.utils.dataset.vision_utils import process_image, process_video
                images=[process_image(cropped_image)]
                
                multi_modal_data = {
                    "image": images
                }
                additional_data = {
                    "multimodal_data": multi_modal_data
                }
                response="""* Step 2 Format: <answer>blond or dark</answer>

Step 2: Now, using the cropped image <image> of the hair (based on your previous bounding box) provided, classify the image as either "blond" or "dark". Provide your classification result in the format <answer>blond or dark</answer>."""
                should_terminate_sequence=False
            else:
                reward=0.0
                response = "Your response don't match format"
                should_terminate_sequence = True
                additional_data={}
        except Exception as e:
            print(e)
            additional_data={}
            response="Your response don't match format"
            should_terminate_sequence = True
            reward=0.0
        
        await self.finalize_interaction(instance_id)
        return should_terminate_sequence, response, reward-1, additional_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        pass

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
