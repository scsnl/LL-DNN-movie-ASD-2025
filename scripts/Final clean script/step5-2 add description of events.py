"""
Step 5-2: Narrative Event Captioning with BLIP-2

This script complements the emotion labeling by generating natural language 
descriptions for the identified video segments. 
Logic:
1. Loads the scene/event boundaries from Step 5.
2. Extracts the middle frame of each segment as a representative keyframe.
3. Uses the BLIP-2 (Flan-T5-XL) model to generate a one-sentence description of the keyframe.
4. Appends descriptions to the event database for qualitative interpretation of 
   model-identified neural events.
"""

import os
import pandas as pd
from pathlib import Path

# Fast test mode bypass
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

if TEST_MODE:
    def _add_dummy_desc(in_csv: str, out_csv: str):
        if not os.path.exists(in_csv): return
        df = pd.read_csv(in_csv)
        df["description"] = "TEST_MODE: dummy caption placeholder."
        df.to_csv(out_csv, index=False)
    _add_dummy_desc("semantic_event_emotions.csv", "semantic_event_emotions_with_desc.csv")
    _add_dummy_desc("visual_scene_emotions.csv", "visual_scene_emotions_with_desc.csv")
    import sys; sys.exit(0)

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/hf_cache")
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BLIP-2 + Flan-T5-XL...")
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def extract_middle_frame(video_path, start_sec, end_sec):
    cap = cv2.VideoCapture(video_path); mid = (start_sec + end_sec) / 2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000); ret, frame = cap.read(); cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else None

def generate_caption(image):
    prompt = "Describe this image in one sentence."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def process_events_csv(csv_path, video_path, output_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path); descriptions = []
    for idx, row in df.iterrows():
        img = extract_middle_frame(video_path, row['start_sec'], row['end_sec'])
        desc = generate_caption(img) if img else "No frame extracted"
        descriptions.append(desc)
    df['description'] = descriptions; df.to_csv(output_path, index=False)

video_path = "DM.mp4"
process_events_csv("semantic_event_emotions.csv", video_path, "semantic_event_emotions_with_desc.csv")
process_events_csv("visual_scene_emotions.csv", video_path, "visual_scene_emotions_with_desc.csv")
