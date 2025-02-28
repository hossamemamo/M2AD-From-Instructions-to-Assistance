from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from .model_interface import ModelInterface
from utils.prompt_handler import PromptHandler

class Mantis_IDEFICS(ModelInterface):
    def __init__(self, device="cuda"):
        self.processor = AutoProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2") # do_image_splitting is False by default
        self.model = AutoModelForVision2Seq.from_pretrained(
            "TIGER-Lab/Mantis-8B-Idefics2",
            device_map="auto"
        )
        self.generation_kwargs = {
            "max_new_tokens": 4,
            "num_beams": 1,
            "do_sample": False
        }

        self.prompt_handler = MantisIdeficsPromptHandler()
        self._supports_interleaved_text_image = True

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        processed_prompt = self.prompt_handler.handle_image_placeholders(prompt, images)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": processed_prompt},
                ]
            }    
        ]
        prompt_model = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_model, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        response = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        return response

class MantisIdeficsPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure
        """
        img_entries = len(images) // 2
        img_tokens = "<image> " * img_entries

        prompt.format(img_placeholder=img_tokens)

        return prompt