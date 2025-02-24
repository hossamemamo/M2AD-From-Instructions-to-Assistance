from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import copy

from . import ModelInterface
from utils.prompt_handler import PromptHandler

class LLaVa_Video(ModelInterface):
    def __init__(self, device="cuda"):
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map, attn_implementation = None)  # Add any other thing you want to pass in llava_model_args
        self.model.eval()

        self.conv_template = "qwen_1_5"
        self.prompt_handler = LLaVa_VideoPromptHandler()
        self._supports_interleaved_text_image = True

    @property
    def supports_interleaved_text_image(self):
        return self._supports_interleaved_text_image

    def predict(self, prompt, images):
        processed_prompt = self.prompt_handler.handle_image_placeholders(prompt, images)
        with torch.no_grad():
            processed_images = self.__process_images(images)
            response = self.__chat_llava_video(processed_prompt, processed_frames[0])
            return response

    def __chat_llava_video(self, prompt, images):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        cont = model.generate(
            input_ids,
            images=images,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=15,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs

    def __process_images(self, images):
        processed_images = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        processed_images = [processed_images]

class LLaVa_VideoPromptHandler(PromptHandler):
    def handle_image_placeholders(self, prompt, images):
        """
        Replace image placeholders in the prompt using the correct prompt structure
        """
        img_entries = len(images) // 2
        img_tokens = "<image> " * img_entries

        prompt.format(img_placeholder=img_tokens)

        return prompt