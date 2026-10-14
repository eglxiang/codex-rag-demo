import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import os

model_path = os.environ.get(
    "QWEN_VL_MODEL_PATH",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

image_path = "outputs/frames/frames/frame_00000025.jpg"

print("Loading model...", flush=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    local_files_only=True
).to("cuda:0").eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True
)

image = Image.open(image_path).convert("RGB")

prompt = (
    "This frame comes from a news report about parliamentary "
    "elections in Morocco. Analyze the image in English. "
    "Use three sections: "
    "1. VISUAL OBSERVATIONS: What is directly visible? "
    "2. CONTEXTUAL INFERENCE: What is the likely setting "
    "given the news context? Explain your reasoning. "
    "3. UNCERTAINTY: What cannot be confirmed from the image? "
    "Do not treat news context as visual proof."
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda:0")

print("Generating...", flush=True)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False
    )

caption = processor.batch_decode(
    output[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)[0]

print("\nCONTEXTUAL VISUAL ANALYSIS:\n")
print(caption)
