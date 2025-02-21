from mantis.models.mfuyu import chat_mfuyu
from transformers import AutoConfig
from mantis.models.mfuyu import MFuyuProcessor, MFuyuForCausalLM

class Mantis_FUYU:
    def __init__(self, device="cuda"):
        config = AutoConfig.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu")
        config.max_length=16000
        config.vocab_size = config._vocab_size

        self.processor = MFuyuProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu")
        self.model = MFuyuForCausalLM.from_pretrained("TIGER-Lab/Mantis-8B-Fuyu", device_map="auto", torch_dtype=torch.bfloat16, config=config)

    def predict(self, prompt, images):
        response, _ = chat_mfuyu(prompt, images, self.model, self.processor)
        return response