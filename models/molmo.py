from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from . import ModelInterface
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

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        pass


class MolmoPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure.
        """
        pass