import os
import pandas as pd
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

#  Hugging Face 

os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/hf_cache")
os.environ["HF_HUB_READ_TIMEOUT"] = "120"

device = "cuda" if torch.cuda.is_available() else "cpu"

#  BLIP-2 + Flan-T5 

print("Loading BLIP-2 + Flan-T5-XL...")
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=os.environ["TRANSFORMERS_CACHE"])
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=os.environ["TRANSFORMERS_CACHE"]
)
model.to(device)
print("Model loaded to:", device)

# 

def extract_middle_frame(video_path, start_sec, end_sec):
    mid_sec = (start_sec + end_sec) / 2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return None

# 

def generate_caption(image):
    prompt = "Describe this image in one sentence."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# 

def process_events_csv(csv_path, video_path, output_path):
    print(f"\nProcessing: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    descriptions = []
    for idx, row in df.iterrows():
        print(f"Processing event {idx + 1}/{len(df)}...")
        image = extract_middle_frame(video_path, row['start_sec'], row['end_sec'])
        if image is not None:
            try:
                desc = generate_caption(image)
                print("Description generated.")
            except Exception as e:
                print(f"Error generating caption at row {idx}: {e}")
                desc = "Error generating description"
        else:
            print("No frame extracted.")
            desc = "No frame extracted"
        descriptions.append(desc)

    df['description'] = descriptions
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

# 

video_path = "DM.mp4"
process_events_csv("semantic_event_emotions.csv", video_path, "semantic_event_emotions_with_desc.csv")
process_events_csv("visual_scene_emotions.csv", video_path, "visual_scene_emotions_with_desc.csv")
