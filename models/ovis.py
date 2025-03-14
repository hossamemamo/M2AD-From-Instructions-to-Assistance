from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoConfig
import gc
import torch
from PIL import Image

from .model_interface import ModelInterface
from utils.prompt_handler import PromptHandler

class Ovis3B(ModelInterface):
    def __init__(self, device="cuda"):
        config = AutoConfig.from_pretrained("AIDC-AI/Ovis1.6-Llama3.2-3B", trust_remote_code=True)
        config.llm_attn_implementation = None

        self.model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Llama3.2-3B",
                                                    torch_dtype=torch.bfloat16,
                                                    multimodal_max_length=8192,
                                                    trust_remote_code=True,
                                                    config=config).cuda()
        self.model.eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

        self._supports_interleaved_text_image = False
        self.prompt_handler = OvisPromptHandler()

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image
    
    def predict(self, prompt, images):
        prompt, concatenated_image = self.prompt_handler.handle_image_placeholders(prompt, images)
        with torch.no_grad():
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(prompt, [concatenated_image])
            
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)

            pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device).detach()]

            gen_kwargs = dict(
                max_new_tokens=15,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=False
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
        return output

    def unload(self):
        for name, param in self.model.named_parameters():
            if param.device == torch.device('cuda'):
                param.data = param.data.to('cpu')
        
        del self.model
        del self.visual_tokenizer
        del self.text_tokenizer
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU cache


class Ovis8B(ModelInterface):
    def __init__(self, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-8B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()
        self.config = AutoConfig.from_pretrained("AIDC-AI/Ovis1.6-Llama3.2-3B")
        self.config.llm_attn_implementation = None
        self.model.eval()
        
        self._supports_interleaved_text_image = False
        self.prompt_handler = OvisPromptHandler()


class OvisPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Handle images by concatenating them together. Prompt will remain untouched.
        """
        prompt_new = f"<image>\n{prompt}"
        # Create a wide image by concatenating N same-height images
        width = 0
        for image in images:
            width = width + image.width
        height = images[0].height

        # Frames will be on the left, manual pages on the right
        concatenated_image = Image.new('RGB', (width, height))
        cur_width = 0
        for i, image in enumerate(images):
            concatenated_image.paste(image, (cur_width, 0))
            cur_width = cur_width + image.width

        return prompt_new, concatenated_image