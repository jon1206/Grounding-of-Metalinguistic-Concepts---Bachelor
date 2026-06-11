import json
import numpy as np
import torch

# Load existing metalinguistic QIDs
with open("metalinguistic_qids.json", "r") as f:
    existing = json.load(f)

# Load extra QIDs
with open("extra_metalinguistic_qids.json", "r") as f:
    extra = json.load(f)

# Merge
all_terms = {**existing, **extra}
print(f"Existing: {len(existing)}, Extra: {len(extra)}, Total: {len(all_terms)}")

# Load full BigGraph
print("Loading BigGraph names...")
with open(r"C:\Users\jon12\BigGraph\wikidata_names.json\wiki_trans_v1_labels.json", "r", encoding="utf-8") as f:
    names = json.load(f)

print("Building QID index...")
qid_to_idx = {}
for i, name in enumerate(names):
    if name.startswith("<http://www.wikidata.org/entity/Q"):
        qid = name.split("/")[-1].rstrip(">")
        qid_to_idx[qid] = i

# Match all terms
word_order = []
indices = []
for term, qid in all_terms.items():
    if qid in qid_to_idx:
        word_order.append(term)
        indices.append(qid_to_idx[qid])

print(f"Terms in BigGraph: {len(word_order)}")

# Extract embeddings
print("Extracting embeddings...")
vectors = np.load(r"C:\Users\jon12\BigGraph\wikidata_vectors.npy\wiki_trans_v1_vec.npy", mmap_mode='r')
filtered = vectors[indices].copy()

torch.save({
    "words": word_order,
    "matrix": torch.tensor(filtered)
}, "biggraph_metalinguistic_en.pt")

print(f"Saved biggraph_metalinguistic_en.pt - shape: {filtered.shape}")

# Save complete mapping
with open("metalinguistic_all_qids.json", "w") as f:
    json.dump(all_terms, f, indent=2)

# Also save word list
with open("metalinguistic_en.txt", "w", encoding="utf-8") as f:
    for w in sorted(word_order):
        f.write(w + "\n")

print(f"\nAll {len(word_order)} terms:")
for w in sorted(word_order):
    print(f"  {w}")