from mantis.models.mfuyu import chat_mfuyu
from transformers import AutoConfig
from mantis.models.mfuyu import MFuyuProcessor, MFuyuForCausalLM

from utils.prompt_handler import PromptHandler

class Mantis_FUYU:
    def __init__(self, device="cuda"):
        config = AutoConfig.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu")
        config.max_length=16000
        config.vocab_size = config._vocab_size

        self.processor = MFuyuProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu")
        self.model = MFuyuForCausalLM.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu", device_map="auto", torch_dtype=torch.bfloat16, config=config)

        self.prompt_handler = MantisFUYUPromptHandler()

    def predict(self, prompt, images):
        processed_prompt = self.prompt_handler.handle_image_placeholders(prompt, images)
        response, _ = chat_mfuyu(processed_prompt, images, self.model, self.processor)
        return response

class MantisFUYUPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure
        """
        img_entries = len(images) // 2
        img_tokens = "<image> " * img_entries

        prompt.format(img_placeholder=img_tokens)

        return prompt