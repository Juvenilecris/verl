import cv2
content="<box>[12, 11, 102, 102]</box>"
content=content.replace(", ", ",")
print(content)
import re
from verl.interactions.qwenvl_utils import smart_resize,reverse_convert_to_original_format
from PIL import Image
img_path="test.jpg"
try:
    box_pattern = re.compile(r"<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>")
    match = box_pattern.search(content)
    print(match)
    if match:
        print("yes, match found")
        reward=1.0
        coords_str = match.groups()
        box_coords = [int(c) for c in coords_str] # -> [x1, y1, x2, y2]
        image = cv2.imread(img_path)
        orig_height = image.shape[0]
        orig_width = image.shape[1]
        new_height, new_width = smart_resize(orig_height, orig_width)
        box_coords=reverse_convert_to_original_format(box_coords,orig_height, orig_width, new_height, new_width)
        print("box_coords:",box_coords)
        assert box_coords[0]+5<box_coords[2]
        assert box_coords[1]+5<box_coords[3]
        left, upper, right, lower = box_coords
        print("left, upper, right, lower:",left, upper, right, lower)
        print("orig_width, orig_height:",orig_width, orig_height)
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
print("should_terminate_sequence, response, reward, additional_data:",should_terminate_sequence, response, reward, additional_data)