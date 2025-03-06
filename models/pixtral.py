import torch
import gc
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from utils.prompt_handler import PromptHandler
import logging

from .model_interface import ModelInterface

class Pixtral(ModelInterface):
    def __init__(self, device="cuda"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        model_id = "mistral-community/pixtral-12b"
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                            torch_dtype=torch.float16,
                                                            device_map='cuda',
                                                            quantization_config=quantization_config)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.prompt_handler = PixtralPromptHandler()
        self._supports_interleaved_text_image = True

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        processed_prompt = self.prompt_handler.handle_image_placeholders(prompt, images)
        with torch.no_grad():
            prompt = self.processor.apply_chat_template(processed_prompt)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(dtype=torch.float16).to(self.model.device)
            generate_ids = self.model.generate(**inputs, max_new_tokens=5)
            output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][-1:]

        return output

    def unload(self):
        for name, param in self.model.named_parameters():
            if param.device == torch.device('cuda'):
                param.data = param.data.to('cpu')
        
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU cache


class PixtralPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure
        """
        parts = prompt.split("{img_placeholder}")
        if len(parts) != 3: # Expect one placeholder for frames and one for pages
            raise ValueError("Invalid number of image placeholders in the prompt.")

        img_entries = [{"type": "image"} for _ in range(len(images) // 2)] # Either 1 frame and 1 page, or 2 frames and 2 pages

        structured_input = [
            {"type": "text", "content": parts[0]}
        ]

        structured_input.extend(img_entries)

        structured_input.append({"type": "text", "content": parts[1]})

        structured_input.extend(img_entries)

        structured_input.append({"type": "text", "content": parts[2]})

        prompt_structured = [
            {"role": "user", "content": structured_input}
            ]

        #logging.info(f"Prompt: {prompt_structured}")

        return prompt_structured
