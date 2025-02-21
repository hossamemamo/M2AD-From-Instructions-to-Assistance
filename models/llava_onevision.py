from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import copy

class LLaVa_OneVision:
    def __init__(self, device="cuda"):
        # Initialize the model
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
        model_name = "llava_qwen"
        device_map = "auto"

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map, attn_implementation=None)
        self.model.eval()

        self.conv_template = "qwen_1_5"
        self.tokenizer.pad_token_id = 151643

    def predict(self, prompt, images):
        # Predict the completion of the prompt
        with torch.no_grad():
            image_sizes, image_tensor = self.__process_images(images)
            response = self.__chat_llava_video(prompt, image_tensor, image_sizes)
            return response

    def __chat_llava_video(self, prompt, images, image_sizes):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

        cont = self.model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            attention_mask=attention_masks,
            do_sample=False,
            temperature=0,
            max_new_tokens=15
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    def __process_images(self, images):
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.model.device) for _image in image_tensor]
        image_size = [[x.size()] for x in image_tensor]
        return image_sizes, image_tensor