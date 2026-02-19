"""
Step 5: Automated Video Scene Segmentation and Emotion Labeling

This script performs AI-based video annotation to interpret the neural events identified by the model. 
Key components:
1. PySceneDetect: Detects visual shot boundaries based on content change.
2. CLIP (Zero-shot): Classifies keyframes into 30 emotion categories (GoEmotions framework).
3. Semantic Clustering: Groups adjacent video frames based on CLIP embedding similarity.
4. Temporal Visualization: Generates colored barcode plots representing the sequence of 
   emotions/scenes during the movie.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# Environment configuration
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

# Define 30 emotion prompts based on GoEmotions
EMOTION_PROMPTS = [
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

if TEST_MODE:
    print("[TEST_MODE] Generating dummy event CSVs.")
    semantic_df = pd.DataFrame({
        "scene_type": ["semantic"] * 3, "scene_index": [1, 2, 3],
        "start_sec": [0.0, 30.0, 60.0], "end_sec": [29.9, 59.9, 89.9],
        "emotion": ["a neutral and uneventful moment"] * 3,
    })
    visual_df = pd.DataFrame({
        "scene_type": ["visual"] * 3, "scene_index": [1, 2, 3],
        "start_sec": [0.0, 30.0, 60.0], "end_sec": [29.9, 59.9, 89.9],
        "emotion": ["a neutral and uneventful moment"] * 3,
    })
    visual_df.to_csv("visual_scene_emotions.csv", index=False)
    semantic_df.to_csv("semantic_event_emotions.csv", index=False)
else:
    import cv2
    import torch
    from PIL import Image
    from tqdm import tqdm
    from sklearn.cluster import AgglomerativeClustering
    from transformers import CLIPProcessor, CLIPModel
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

    def extract_clip_embeddings(images):
        features = []
        for img in images:
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
            norm_feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(norm_feat.cpu().numpy().squeeze())
        return np.vstack(features)

    def classify_emotion(image):
        inputs = clip_processor(text=EMOTION_PROMPTS, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
        return EMOTION_PROMPTS[np.argmax(probs)], probs

    def cluster_embeddings(embeddings, times, threshold=0.6):
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage="average")
        labels = clustering.fit_predict(embeddings)
        segments = []
        current = labels[0]; start = times[0]
        for i in range(1, len(labels)):
            if labels[i] != current:
                segments.append((start, times[i - 1]))
                start = times[i]; current = labels[i]
        segments.append((start, times[-1]))
        return segments

    def detect_visual_scenes(video_path, threshold=12.0):
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        return [(s.get_seconds(), e.get_seconds()) for s, e in scene_manager.get_scene_list()]

    def extract_keyframe(video_path, start_sec, end_sec):
        cap = cv2.VideoCapture(video_path)
        mid = (start_sec + end_sec) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
        ret, frame = cap.read()
        cap.release()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else None

    def process_video_events(video_path):
        print("📽 Detecting visual scenes...")
        visual_scenes = detect_visual_scenes(video_path)
        visual_records = []
        for i, (start, end) in enumerate(visual_scenes):
            img = extract_keyframe(video_path, start, end)
            if img is not None:
                emo_label, _ = classify_emotion(img)
                visual_records.append({"scene_type": "visual", "scene_index": i + 1, "start_sec": round(start, 2), "end_sec": round(end, 2), "emotion": emo_label})

        print("🧠 Detecting semantic events...")
        times, images = extract_frames(video_path)
        clip_feats = extract_clip_embeddings(images)
        semantic_scenes = cluster_embeddings(clip_feats, times, threshold=0.6)
        semantic_records = []
        for i, (start, end) in enumerate(semantic_scenes):
            img = extract_keyframe(video_path, start, end)
            if img is not None:
                emo_label, _ = classify_emotion(img)
                semantic_records.append({"scene_type": "semantic", "scene_index": i + 1, "start_sec": round(start, 2), "end_sec": round(end, 2), "emotion": emo_label})

        visual_df = pd.DataFrame(visual_records); semantic_df = pd.DataFrame(semantic_records)
        visual_df.to_csv("visual_scene_emotions.csv", index=False)
        semantic_df.to_csv("semantic_event_emotions.csv", index=False)
        return visual_df, semantic_df

    video_path = "DM.mp4"
    if os.path.exists(video_path):
        visual_df, semantic_df = process_video_events(video_path)

# Visualization
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

def draw_colored_events(df, title, output_path):
    all_emotions = df['emotion'].unique()
    appeared_emotions = [e for e in emotion_category_map if e in all_emotions]
    emotion_groups = {'positive': [], 'neutral': [], 'negative': []}
    for emo in appeared_emotions: emotion_groups[emotion_category_map[emo]].append(emo)
    
    pos_p = sns.color_palette("Oranges", len(emotion_groups['positive']))
    neu_p = sns.color_palette("Greens", len(emotion_groups['neutral']))
    neg_p = sns.color_palette("Blues", len(emotion_groups['negative']))
    
    color_map = {}
    for i, e in enumerate(emotion_groups['positive']): color_map[e] = pos_p[i]
    for i, e in enumerate(emotion_groups['neutral']): color_map[e] = neu_p[i]
    for i, e in enumerate(emotion_groups['negative']): color_map[e] = neg_p[i]

    fig, ax = plt.subplots(figsize=(14, 2.2))
    for _, row in df.iterrows():
        start = row['start_sec']; width = row['end_sec'] - row['start_sec']
        ax.broken_barh([(start, width)], (0, 5), facecolors=color_map.get(row['emotion'], "#CCCCCC"), edgecolor='black', linewidth=0.7)

    ax.set_xlim(0, df['end_sec'].max() + 2); ax.set_ylim(0, 5); ax.set_xlabel("Time (seconds)")
    ax.set_title(title, weight='bold'); ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()

if os.path.exists("visual_scene_emotions.csv"):
    draw_colored_events(pd.read_csv("visual_scene_emotions.csv"), "Visual Scene Segments", "visual_blocks.png")
    draw_colored_events(pd.read_csv("semantic_event_emotions.csv"), "Semantic Event Segments", "semantic_blocks.png")
