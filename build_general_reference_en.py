"""
RECONSTRUCTION: the original English label-matching script was not
preserved. This is a faithful reconstruction with logic identical to
build_general_reference_da.py, using @en labels and the English 20k
frequency list (first20hours/google-10000-english, 20k.txt).

The original script's output, biggraph_embeddings.pt (14,070 words), IS
included in this repository. To verify the reconstruction, run this script
(after editing the machine-specific paths to the raw BigGraph dump) and
compare:

    import torch
    old = torch.load("biggraph_embeddings.pt", weights_only=False)
    new = torch.load("biggraph_embeddings_reconstructed.pt", weights_only=False)
    print(len(old["words"]), len(new["words"]),
          set(old["words"]) == set(new["words"]))

The output filename is deliberately different so the original is never
overwritten.
"""

import json
import numpy as np
import torch

# Machine-specific paths to the raw pretrained BigGraph WikiData dump
# (see README for download location) — edit before running.
LABELS_PATH = r"C:\Users\jon12\BigGraph\wikidata_names.json\wiki_trans_v1_labels.json"
VECTORS_PATH = r"C:\Users\jon12\BigGraph\wikidata_vectors.npy\wiki_trans_v1_vec.npy"

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    names = json.load(f)

with open("20k.txt", "r", encoding="utf-8") as f:
    english_words = set(line.strip().lower() for line in f if line.strip())

matches = {}
for i, name in enumerate(names):
    if name.endswith('@en'):
        word = name.split('"')[1]
        if ' ' not in word and word.lower() in english_words:
            key = word.lower()
            if key not in matches:
                matches[key] = i

print(f"Matched: {len(matches)}")

vectors = np.load(VECTORS_PATH, mmap_mode='r')

word_order = list(matches.keys())
indices = [matches[w] for w in word_order]
filtered_vectors = vectors[indices].copy()

print(f"Filtered shape: {filtered_vectors.shape}")

torch.save({
    "words": word_order,
    "matrix": torch.tensor(filtered_vectors)
}, "biggraph_embeddings_reconstructed.pt")

print("Saved biggraph_embeddings_reconstructed.pt")
