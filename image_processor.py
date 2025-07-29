# image_processor.py

import os
from PIL import Image
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import torch

# Load model once
disable_torch_init()
model_path = "LLaVA/weights/llava-v1.5-7b"  # adjust this to your actual model dir
tokenizer, model, processor, context_len = load_pretrained_model(
    model_path, model_base=None, model_name="llava-v1.5-7b"
)

def analyze_image_with_llava(image_path: str, prompt: str = None):
    if prompt is None:
        prompt = """You are a Vision-Language Model assistant.
Your task is to detect any visual signs that might affect the user's cognitive load.
Analyze the image for:
- Distractions in the background (other screens, phone, clutter)
- User’s posture (slouching, leaning, alert)
- Facial signs of strain, confusion, or engagement
- Visible multi-tasking (holding phone, other apps open)

Output a short summary:
- Environment distractions: [describe]
- User posture: [describe]
- Facial cues: [describe]
- Any other factor that might impact focus."""

    image = Image.open(image_path).convert("RGB")

    # Setup conversation template
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    image_tensor = process_images([image], processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX=32000, return_tensors="pt").unsqueeze(0).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return output
