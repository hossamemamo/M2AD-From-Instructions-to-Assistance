from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

class Mantis_IDEFICS:
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

    def predict(self, prompt, images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_tmp},
                ]
            }    
        ]
        prompt_model = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        response = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        return response

