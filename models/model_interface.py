from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def predict(self, prompt, images):
        pass

    @abstractmethod
    def supports_interleaved_text_image(self):
        pass

    @abstractmethod
    def unload(self):
        pass