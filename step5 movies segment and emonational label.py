import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from transformers import CLIPProcessor, CLIPModel
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import warnings
warnings.filterwarnings("ignore")  #  PySceneDetect 


# ===  ===

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ===  GoEmotions +  ===

goemotions_prompts = [
    "a moment of admiration", "an amusing scene", "a moment of anger", "an annoying scene",
    "a moment of approval", "a caring and gentle moment", "a moment of confusion",
    "a curious and exploring scene", "a scene of strong desire", "a disappointing moment",
    "a disapproving expression", "a disgusting moment", "an embarrassing situation",
    "an exciting and energetic scene", "a fearful or scary moment", "a moment of gratitude",
    "a grieving and sorrowful moment", "a joyful and happy moment", "a loving and affectionate moment",
    "a nervous or anxious scene", "an optimistic and uplifting moment", "a proud and confident moment",
    "a moment of realization or discovery", "a relieving and calming moment",
    "a remorseful or regretful moment", "a sad and emotional scene",
    "a surprising and unexpected moment",
    "a neutral and uneventful moment", "a calm and quiet scene", "a transition with little emotion"
]

# === 1 ===

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_times, frame_images = [], []
    for t in range(int(duration)):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_images.append(Image.fromarray(rgb))
            frame_times.append(t)
    cap.release()
    return frame_times, frame_images

# === CLIP  ===

def extract_clip_embeddings(images):
    features = []
    for img in images:
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
        norm_feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(norm_feat.cpu().numpy().squeeze())
    return np.vstack(features)

# === CLIP + softmax ===

def classify_emotion(image):
    inputs = clip_processor(text=goemotions_prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
    return goemotions_prompts[np.argmax(probs)], probs

# ===  CLIP  ===

def cluster_embeddings(embeddings, times, threshold=1.0):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage="average")
    labels = clustering.fit_predict(embeddings)
    segments = []
    current = labels[0]
    start = times[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, times[i - 1]))
            start = times[i]
            current = labels[i]
    segments.append((start, times[-1]))
    return segments

# ===  PySceneDetect  ===

def detect_visual_scenes(video_path, threshold=12.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_manager.get_scene_list()]

# ===  ===

def extract_keyframe(video_path, start_sec, end_sec):
    cap = cv2.VideoCapture(video_path)
    mid = (start_sec + end_sec) / 2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    else:
        return None

# ===  +   +  ===

def process_video_events(video_path):
    print(" Detecting visual scenes...")
    visual_scenes = detect_visual_scenes(video_path)
    visual_records = []
    for i, (start, end) in enumerate(visual_scenes):
        img = extract_keyframe(video_path, start, end)
        if img is None:
            continue
        emo_label, _ = classify_emotion(img)
        visual_records.append({
            "scene_type": "visual",
            "scene_index": i + 1,
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "emotion": emo_label
        })

    print(" Detecting semantic events...")
    times, images = extract_frames(video_path)
    clip_feats = extract_clip_embeddings(images)
    semantic_scenes = cluster_embeddings(clip_feats, times, threshold=0.6)
    semantic_records = []
    for i, (start, end) in enumerate(semantic_scenes):
        img = extract_keyframe(video_path, start, end)
        if img is None:
            continue
        emo_label, _ = classify_emotion(img)
        semantic_records.append({
            "scene_type": "semantic",
            "scene_index": i + 1,
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "emotion": emo_label
        })

    #  CSV

    visual_df = pd.DataFrame(visual_records)
    semantic_df = pd.DataFrame(semantic_records)
    visual_df.to_csv("visual_scene_emotions.csv", index=False)
    semantic_df.to_csv("semantic_event_emotions.csv", index=False)

    print(" Results saved to 'visual_scene_emotions.csv' and 'semantic_event_emotions.csv'")
    return visual_df, semantic_df

# ===  ===

if __name__ == "__main__":
    video_path = "DM.mp4"  # 

    process_video_events(video_path)
    

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ===  ===

visual_df = pd.read_csv("visual_scene_emotions.csv")
semantic_df = pd.read_csv("semantic_event_emotions.csv")

# === prompt  ===

emotion_category_map = {
    "a moment of admiration": "positive", "an amusing scene": "positive", "a moment of approval": "positive",
    "a caring and gentle moment": "positive", "an exciting and energetic scene": "positive",
    "a moment of gratitude": "positive", "a joyful and happy moment": "positive", "a loving and affectionate moment": "positive",
    "an optimistic and uplifting moment": "positive", "a proud and confident moment": "positive",
    "a relieving and calming moment": "positive", "a surprising and unexpected moment": "positive",
    "a curious and exploring scene": "neutral", "a moment of confusion": "neutral", "a scene of strong desire": "neutral",
    "a moment of realization or discovery": "neutral", "a neutral and uneventful moment": "neutral",
    "a calm and quiet scene": "neutral", "a transition with little emotion": "neutral",
    "a moment of anger": "negative", "an annoying scene": "negative", "a disappointing moment": "negative",
    "a disapproving expression": "negative", "a disgusting moment": "negative", "an embarrassing situation": "negative",
    "a fearful or scary moment": "negative", "a grieving and sorrowful moment": "negative",
    "a nervous or anxious scene": "negative", "a remorseful or regretful moment": "negative",
    "a sad and emotional scene": "negative"
}

# ===  ===

all_emotions = pd.concat([visual_df['emotion'], semantic_df['emotion']]).unique()
appeared_emotions = [e for e in emotion_category_map if e in all_emotions]

# ===  ===

emotion_groups = {'positive': [], 'neutral': [], 'negative': []}
for emo in appeared_emotions:
    cat = emotion_category_map[emo]
    emotion_groups[cat].append(emo)

# === ===

positive_palette = sns.color_palette("Oranges", len(emotion_groups['positive']))
neutral_palette  = sns.color_palette("Greens", len(emotion_groups['neutral']))
negative_palette = sns.color_palette("Blues", len(emotion_groups['negative']))

# ===  emotion  ===

emotion_color_map = {}
for i, emo in enumerate(emotion_groups['positive']):
    emotion_color_map[emo] = positive_palette[i]
for i, emo in enumerate(emotion_groups['neutral']):
    emotion_color_map[emo] = neutral_palette[i]
for i, emo in enumerate(emotion_groups['negative']):
    emotion_color_map[emo] = negative_palette[i]

# ===  ===

def draw_colored_events(df, title, output_path):
    fig, ax = plt.subplots(figsize=(14, 2.2))
    for _, row in df.iterrows():
        emo = row['emotion']
        start = row['start_sec']
        width = row['end_sec'] - row['start_sec']
        color = emotion_color_map.get(emo, "#CCCCCC")

        ax.broken_barh([(start, width)], (0, 5),
                       facecolors=color, edgecolor='black', linewidth=0.7)

    ax.set_xlim(0, df['end_sec'].max() + 2)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved: {output_path}")

# === ===

def draw_emotion_legend(save_path="legend_actual_emotions.png"):
    patches = []
    for emo in appeared_emotions:
        patches.append(mpatches.Patch(color=emotion_color_map[emo], label=emo))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", frameon=False, ncol=2, fontsize=9, title="Appeared Emotions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Saved: {save_path}")

# ===  ===

draw_colored_events(visual_df, "Visual Scene Segments (Soft Morandi Colors)", "visual_blocks_morandi_filtered.png")
draw_colored_events(semantic_df, "Semantic Event Segments (Soft Morandi Colors)", "semantic_blocks_morandi_filtered.png")
draw_emotion_legend("legend_actual_emotions.png")
