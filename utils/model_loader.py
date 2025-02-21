from models import MODEL_REGISTRY

def load_model(model_name, device="cuda"):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(device=device)