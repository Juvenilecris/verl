import math
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]

def reverse_convert_to_original_format(bbox_new, orig_height, orig_width, new_height, new_width):
    """
    从修改后的坐标恢复原始坐标值。
    
    参数:
    - bbox_new: 修改后的边界框坐标 [x1_new, y1_new, x2_new, y2_new]
    - orig_height: 原始图像的高度
    - orig_width: 原始图像的宽度
    - new_height: 修改后的图像高度
    - new_width: 修改后的图像宽度
    
    返回:
    - 原始坐标 [x1, y1, x2, y2]
    """
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1_new, y1_new, x2_new, y2_new = bbox_new
    
    # 反向计算原始坐标
    x1 = round(x1_new / scale_w)
    y1 = round(y1_new / scale_h)
    x2 = round(x2_new / scale_w)
    y2 = round(y2_new / scale_h)
    
    # 确保原始坐标在原始图像范围内
    x1 = max(0, min(x1, orig_width - 1))
    y1 = max(0, min(y1, orig_height - 1))
    x2 = max(0, min(x2, orig_width - 1))
    y2 = max(0, min(y2, orig_height - 1))
    
    return [x1, y1, x2, y2]
