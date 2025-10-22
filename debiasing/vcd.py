import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os
from typing import List, Dict, Tuple

class VCDProcessor:
    def __init__(self, model_id: str = "qwen-vl/qwen2.5-vl-3b-instruct"):
        print("Initializing VCDProcessor: Loading model and processor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        # 关键：为批量处理设置 padding
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        print("Initialization complete.")
        
    def _get_logits_from_vcd_batch(
        self, 
        normal_image_inputs, 
        contrastive_image_inputs, 
        alpha, beta, 
        max_new_tokens=150
    ) -> Tuple[List[str], torch.Tensor]:
        """
        内部方法：执行 VCD 批量生成循环。
        """
        batch_size = normal_image_inputs['input_ids'].shape[0]
        
        # 初始输入和注意力掩码
        input_ids = normal_image_inputs['input_ids']
        attention_mask = normal_image_inputs['attention_mask']
        pixel_values_normal = normal_image_inputs['pixel_values']
        pixel_values_contrastive = contrastive_image_inputs['pixel_values']
        
        generated_ids = input_ids.clone()
        final_scores = None

        self.model.eval()
        
        # 跟踪每个序列是否已完成
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                # 1. 批量获取正常图像的 logits
                outputs_normal = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values_normal
                )
                scores_normal = outputs_normal.logits[:, -1, :]

                # 2. 批量获取对比图像的 logits
                outputs_contrastive = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values_contrastive
                )
                scores_contrastive = outputs_contrastive.logits[:, -1, :]

                # 3. 批量应用 VCD 公式
                contrastive_scores = (1 + alpha) * scores_normal - alpha * scores_contrastive

                # 4. 批量应用自适应约束
                original_probs = torch.nn.functional.softmax(scores_normal, dim=-1)
                mask = original_probs < (beta * original_probs.max(dim=-1, keepdim=True)[0])
                contrastive_scores[mask] = -float('Inf')

                # 5. 贪心解码
                next_token_ids = torch.argmax(contrastive_scores, dim=-1)
                
                # 如果某个序列已经生成了 EOS token，则后续用 pad token 填充
                next_token_ids = next_token_ids * unfinished_sequences + self.processor.tokenizer.pad_token_id * (1 - unfinished_sequences)
                
                # 6. 附加新 token
                generated_ids = torch.cat([generated_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                
                # 更新 attention_mask
                attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)

                # 更新 input_ids 以供下一步使用
                input_ids = next_token_ids.unsqueeze(-1)
                
                # 8. 检查停止条件
                unfinished_sequences = unfinished_sequences.mul(
                    (next_token_ids != self.processor.tokenizer.eos_token_id).long()
                )
                
                # 如果所有序列都已完成，则提前退出
                if unfinished_sequences.max() == 0:
                    break

        final_scores = contrastive_scores.clone().cpu()
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts, final_scores

    def process_batch(self, batch_data: List[Dict], alpha: float = 0.5, beta: float = 0.1):
        """
        外部接口：处理一批数据。
        """
        normal_images = [Image.open(item["normal_path"]) for item in batch_data]
        contrastive_images = [Image.open(item["contrastive_path"]) for item in batch_data]
        prompts = [item["prompt"] for item in batch_data]

        texts = []
        for p in prompts:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        # 关键：使用 padding=True 来处理批量中不同长度的文本
        normal_inputs = self.processor(text=texts, images=normal_images, return_tensors="pt", padding=True).to(self.device, dtype=self.model.dtype)
        contrastive_inputs = self.processor(text=texts, images=contrastive_images, return_tensors="pt", padding=True).to(self.device, dtype=self.model.dtype)
        
        generated_texts, final_logits = self._get_logits_from_vcd_batch(
            normal_inputs, contrastive_inputs, alpha=alpha, beta=beta
        )

        return generated_texts, final_logits

# --- 主程序：演示如何使用批量处理 ---
if __name__ == "__main__":
    vcd_processor = VCDProcessor("/data/wangnn/models/Qwen2.5-VL-3B-Instruct")

    data_points = [
        {"normal_path": "path/to/normal_image_1.jpg", "contrastive_path": "path/to/contrastive_image_1.jpg", "prompt": "详细描述图片内容。"},
        {"normal_path": "path/to/normal_image_2.jpg", "contrastive_path": "path/to/contrastive_image_2.jpg", "prompt": "描述图中的主要物体。"},
        # 可以添加更多数据...
    ]
    
    print("\nStarting batch processing with VCD...")
    
    # 一次性处理所有数据
    generated_texts, final_logits = vcd_processor.process_batch(data_points, alpha=0.5, beta=0.1)

    print("\n--- Batch VCD Results ---")
    for i, text in enumerate(generated_texts):
        print(f"Result for item {i+1}:")
        print(text.strip())
        print("-" * 20)
    
    print(f"Final batch logits shape: {final_logits.shape}")