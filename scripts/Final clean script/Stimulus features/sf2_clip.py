"""
CLIP softmax over the 30 emotion prompts of step 5, for every frame.

Scoring follows step 5: CLIPProcessor(text=prompts, images=..., padding=True),
logits_per_image, softmax over prompts. The softmax is not calibrated; later
steps use only the argmax label and the order of the lexicon norms.
With TEST_MODE=1 the model is not loaded and seeded random probabilities are
written instead.

Outputs: clip_probs_<movie>.csv (per frame), clip_probs_<movie>_tr.csv (per TR)
"""
import numpy as np
import pandas as pd

from sf_common import (CACHE, CLIP_CHECKPOINT, FRAMES_PER_TR, MOVIES, OUT, SEED, TEST_MODE, TR,
                       ensure_dirs, load_prompts)

BATCH = 32


def clip_probs(man, fdir, prompts):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_CHECKPOINT).to(device).eval()
    # use_fast=False: the slow image processor resizes as in step 5
    processor = CLIPProcessor.from_pretrained(CLIP_CHECKPOINT, use_fast=False)
    probs = np.zeros((len(man), len(prompts)))
    for i in range(0, len(man), BATCH):
        chunk = man.iloc[i:i + BATCH]
        imgs = [Image.open(fdir / f).convert("RGB") for f in chunk["file"]]
        inputs = processor(text=prompts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs[i:i + len(chunk)] = model(**inputs).logits_per_image.softmax(dim=1).float().cpu().numpy()
    return probs


def placeholder_probs(n, k, rng):
    z = rng.normal(scale=2.0, size=(n, k))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ensure_dirs()
    prompts = load_prompts()
    rng = np.random.default_rng(SEED)
    if TEST_MODE:
        print("TEST_MODE: placeholder CLIP probabilities")
    else:
        import torch
        torch.manual_seed(SEED)
    for m in MOVIES:
        man = pd.read_csv(OUT / f"frames_{m}.csv")
        probs = (placeholder_probs(len(man), len(prompts), rng) if TEST_MODE
                 else clip_probs(man, CACHE / f"frames_{m}", prompts))
        cols = [f"p{j:02d}" for j in range(len(prompts))]
        df = pd.DataFrame(probs, columns=cols)
        df.insert(0, "frame_index", man["frame_index"].to_numpy())
        df.insert(1, "tr_index", man["tr_index"].to_numpy())
        df.insert(2, "time_sec", man["target_time_sec"].to_numpy())
        df["argmax_prompt"] = [prompts[j] for j in probs.argmax(axis=1)]
        df["max_prob"] = probs.max(axis=1)
        df.to_csv(OUT / f"clip_probs_{m}.csv", index=False)

        n_tr = int(man["tr_index"].max()) + 1
        trp = probs.reshape(n_tr, FRAMES_PER_TR, len(prompts)).mean(axis=1)
        tdf = pd.DataFrame(trp, columns=cols)
        tdf.insert(0, "tr_index", np.arange(n_tr))
        tdf.insert(1, "time_sec", (np.arange(n_tr) + 0.5) * TR)
        tdf["argmax_prompt"] = [prompts[j] for j in trp.argmax(axis=1)]
        tdf["max_prob"] = trp.max(axis=1)
        tdf.to_csv(OUT / f"clip_probs_{m}_tr.csv", index=False)
        print(f"{m}: {len(man)} frames, {tdf['argmax_prompt'].nunique()} distinct TR labels")


if __name__ == "__main__":
    main()
