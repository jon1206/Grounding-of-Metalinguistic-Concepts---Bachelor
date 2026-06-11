import json
import numpy as np
import torch

# Load QID mappings
with open("metalinguistic_all_qids.json", "r") as f:
    en_terms = json.load(f)

with open("metalinguistic_en_to_da.json", "r", encoding="utf-8") as f:
    en_to_da = json.load(f)

# Load BigGraph QID index
print("Loading BigGraph names...")
with open(r"C:\Users\jon12\BigGraph\wikidata_names.json\wiki_trans_v1_labels.json", "r", encoding="utf-8") as f:
    names = json.load(f)

print("Building QID index...")
qid_to_idx = {}
for i, name in enumerate(names):
    if name.startswith("<http://www.wikidata.org/entity/Q"):
        qid = name.split("/")[-1].rstrip(">")
        qid_to_idx[qid] = i

# Use the SAME QID embeddings as English (same concepts, just Danish labels)
vectors = np.load(r"C:\Users\jon12\BigGraph\wikidata_vectors.npy\wiki_trans_v1_vec.npy", mmap_mode='r')

da_words = []
indices = []
for en_term, da_term in en_to_da.items():
    qid = en_terms[en_term]
    if qid in qid_to_idx:
        da_words.append(da_term)
        indices.append(qid_to_idx[qid])

filtered = vectors[indices].copy()

torch.save({
    "words": da_words,
    "matrix": torch.tensor(filtered)
}, "biggraph_metalinguistic_da.pt")

print(f"Saved biggraph_metalinguistic_da.pt - shape: {filtered.shape}")
print(f"Terms: {len(da_words)}")