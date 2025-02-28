from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoConfig

from .model_interface import ModelInterface
from utils.prompt_handler import PromptHandler

class Ovis(ModelInterface):
    def __init__(self, device="cuda"):
        config = AutoConfig.from_pretrained("AIDC-AI/Ovis1.6-Llama3.2-3B")
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
        concatenated_image = self.prompt_handler.handle_image_placeholders(prompt, images)

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(prompt, [concatenated_image])
        
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)

        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device).detach()]

        gen_kwargs = dict(
            max_new_tokens=2,
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

class OvisPromptHandler(PromptHandler):
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