from mantis.models.mfuyu import chat_mfuyu
from transformers import AutoConfig
from mantis.models.mfuyu import MFuyuProcessor, MFuyuForCausalLM
import torch
import gc

from .model_interface import ModelInterface
from utils.prompt_handler import PromptHandler

class Mantis_FUYU(ModelInterface):
    def __init__(self, device="cuda"):
        config = AutoConfig.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu")
        config.max_length=16000
        config.vocab_size = config._vocab_size

        self.processor = MFuyuProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu")
        self.model = MFuyuForCausalLM.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu", device_map="auto", torch_dtype=torch.bfloat16, config=config)

        self.prompt_handler = MantisFUYUPromptHandler()
        self._supports_interleaved_text_image = True

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        processed_prompt = self.prompt_handler.handle_image_placeholders(prompt, images)
        response, _ = chat_mfuyu(processed_prompt, images, self.model, self.processor)
        return response

    def unload(self):
        for name, param in self.model.named_parameters():
            if param.device == torch.device('cuda'):
                param.data = param.data.to('cpu')
        
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU cache

class MantisFUYUPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure
        """
        img_entries = len(images) // 2
        img_tokens = "<image> " * img_entries

        prompt = prompt.format(img_placeholder=img_tokens)

        return prompt