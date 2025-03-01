from transformers import BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch

from .model_interface import ModelInterface
from utils.prompt_handler import PromptHandler

class Qwen2VL(ModelInterface):
    def __init__(self, device="cuda"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.model.eval()

        self.prompt_handler = Qwen2VLPromptHandler()
        self._supports_interleaved_text_image = True

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        processed_prompt = self.prompt_handler.handle_image_placeholders(prompt, images)
        with torch.no_grad():
            text = self.processor.apply_chat_template(processed_prompt, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to('cuda')

            # Batch Inference
            output_ids = self.model.generate(**inputs, max_new_tokens=10)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return output

class Qwen2VLPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure
        """
        parts = prompt.split("{img_placeholder}")
        if len(parts) != 3: # Expect one placeholder for frames and one for pages
            raise ValueError("Invalid number of image placeholders in the prompt.")

        img_entries = [{"type": "image"} for _ in range(len(images) // 2)] # Either 1 frame and 1 page, or 2 frames and 2 pages

        structured_input = [
            {"type": "text", "text": parts[0]}
        ]

        structured_input.extend(img_entries)

        structured_input.append({"type": "text", "text": parts[1]})

        structured_input.extend(img_entries)

        structured_input.append({"type": "text", "text": parts[2]})

        prompt_structured = [
                    {"role": "user", "content": structured_input}
                    ]

        return prompt_structured