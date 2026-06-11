import json
import numpy as np
import torch

with open(r"C:\Users\jon12\BigGraph\wikidata_names.json\wiki_trans_v1_labels.json", "r", encoding="utf-8") as f:
    names = json.load(f)

with open("danish_20k.txt", "r", encoding="utf-8") as f:
    danish_words = set(line.strip().lower() for line in f if line.strip())

matches = {}
for i, name in enumerate(names):
    if name.endswith('@da'):
        word = name.split('"')[1]
        if ' ' not in word and word.lower() in danish_words:
            key = word.lower()
            if key not in matches:
                matches[key] = i

print(f"Matched: {len(matches)}")

vectors = np.load(r"C:\Users\jon12\BigGraph\wikidata_vectors.npy\wiki_trans_v1_vec.npy", mmap_mode='r')

word_order = list(matches.keys())
indices = [matches[w] for w in word_order]
filtered_vectors = vectors[indices].copy()

print(f"Filtered shape: {filtered_vectors.shape}")

torch.save({
    "words": word_order,
    "matrix": torch.tensor(filtered_vectors)
}, "biggraph_danish_embeddings.pt")

print("Saved biggraph_danish_embeddings.pt")