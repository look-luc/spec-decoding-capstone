"""
Load models for speculative decoding tasks.
"""
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_model(model_name: str, device: str = "auto", load_in_4bit: bool = True):
    """
    Load a HuggingFace model and tokenizer with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model name or local path
        device: "cuda", "mps", "cpu", or "auto"
        load_in_4bit: Whether to load the model in 4-bit via bitsandbytes (saves massive VRAM)

    Returns:
        tuple of (model, tokenizer)
    """
    dev = _resolve_device(device)
    logger.info(f"Loading {model_name} on device={dev} (4-bit: {load_in_4bit})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = None
    if load_in_4bit and dev.type == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if dev.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16 if quantization_config is None else None,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if dev.type == "mps" else torch.float32,
            trust_remote_code=True,
        )
        model = model.to(dev)

    model.eval()
    return model, tokenizer
