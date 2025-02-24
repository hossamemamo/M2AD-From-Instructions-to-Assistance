from abc import ABC, abstractmethod

from .llava_onevision import LLaVa_OneVision
from .llava_video import LLaVa_Video

MODEL_REGISTRY = {
    "LLAVA-OneVision": LLaVa_OneVision,
    "LLAVA-Video": LLaVa_Video
}

class ModelInterface(ABC):
    @abstractmethod
    def predict(self, prompt, images):
        pass

    @abstractmethod
    def supports_interleaved_text_image(self):
        pass