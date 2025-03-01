from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch

from .model_interface import ModelInterface
from utils.prompt_handler import PromptHandler

class Molmo(ModelInterface):
    def __init__(self, device="cuda"):
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        self._supports_interleaved_text_image = False # To be used in exp files for the choice of the correct prompt. Interleaved will use <img> tokens, non-interleaved will say left/right in the image.
        self.prompt_handler = MolmoPromptHandler()

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        # We have the non-interleaved text-image prompt here, so we want to process the images by putting them side-by-side, and then process the input normally.
        concatenated_image = self.prompt_handler.handle_image_placeholders(prompt, images)
        with torch.no_grad():
            inputs = self.processor.process(
                images=concatenated_image,
                text=prompt
            )

            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )
            
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_text

class MolmoPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Handle images by concatenating them together. Prompt will remain untouched.
        """
        n_images = len(images)

        # Create a wide image by concatenating N same-height images
        width = images[0].width * n_images
        height = images[0].height

        # Frames will be on the left, manual pages on the right
        concatenated_image = Image.new('RGB', (width, height))

        for i, image in enumerate(images):
            concatenated_image.paste(image, (i * image.width, 0))

        return concatenated_image