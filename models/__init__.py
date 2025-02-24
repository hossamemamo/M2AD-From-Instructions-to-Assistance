from abc import ABC, abstractmethod

from .llava_onevision import LLaVa_OneVision
from .llava_video import LLaVa_Video
from .mantis_fuyu import Mantis_FUYU
from .mantis_idefics import Mantis_IDEFICS
from .pixtral import Pixtral
from .qwen2_vl import Qwen2VL
from .molmo import Molmo
from .ovis import Ovis

MODEL_REGISTRY = {
    "LLAVA-OneVision": LLaVa_OneVision,
    "LLAVA-Video": LLaVa_Video,
    "Mantis-FUYU": Mantis_FUYU,
    "Mantis-IDEFICS": Mantis_IDEFICS,
    "Pixtral": Pixtral,
    "Qwen2VL": Qwen2VL,
    "Molmo": Molmo,
    "Ovis": Ovis
}

def load_model(model_name, device="cuda"):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry. Available models {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](device=device)

class ModelInterface(ABC):
    @abstractmethod
    def predict(self, prompt, images):
        pass

    @abstractmethod
    def supports_interleaved_text_image(self):
        pass