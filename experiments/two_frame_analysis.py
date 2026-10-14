import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import os

model_path = os.environ.get(
    "QWEN_VL_MODEL_PATH",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

image_path_30 = "outputs/frames/frames/frame_00000750.jpg"
image_path_31 = "outputs/frames/frames/frame_00000775.jpg"

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

image30 = Image.open(image_path_30).convert("RGB")
image31 = Image.open(image_path_31).convert("RGB")

prompt = """
You are given two consecutive video frames:
Frame 1 at 30 seconds and Frame 2 at 31 seconds.

Compare the two images in English.

1. SCENE CONTINUITY:
Are they from the same scene? What evidence supports this?

2. OBJECT CORRESPONDENCE:
Which people or objects appear in both frames?
Describe their visual attributes.

3. STATE CHANGES:
What observable changes occur between the frames?
Distinguish changes in image position from confirmed motion.

4. UNCERTAINTY:
What cannot be determined from only these two frames?

Do not invent actions, identities, or events.
"""

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Frame 1 (30 seconds):"},
        {"type": "image", "image": image30},
        {"type": "text", "text": "Frame 2 (31 seconds):"},
        {"type": "image", "image": image31},
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
