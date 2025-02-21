from .llava_onevision import LLaVa_OneVision
from .llava_video import LLaVa_Video

MODEL_REGISTRY = {
    "LLAVA-OneVision": LLaVa_OneVision,
    "LLAVA-Video": LLaVa_Video
}