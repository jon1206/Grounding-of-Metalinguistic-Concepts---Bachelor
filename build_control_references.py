import torch

# English control BigGraph
bg_en = torch.load("biggraph_embeddings.pt", weights_only=False)
with open("en_138.txt", "r") as f:
    control_en = set(line.strip().lower() for line in f)

en_words = bg_en["words"]
en_matrix = bg_en["matrix"]
mask = [i for i, w in enumerate(en_words) if w in control_en]
torch.save({
    "words": [en_words[i] for i in mask],
    "matrix": en_matrix[mask]
}, "biggraph_control_en138.pt")
print(f"EN control BigGraph: {len(mask)} words")

# Danish control BigGraph
bg_da = torch.load("biggraph_danish_embeddings.pt", weights_only=False)
with open("da_122.txt", "r", encoding="utf-8") as f:
    control_da = set(line.strip().lower() for line in f)

da_words = bg_da["words"]
da_matrix = bg_da["matrix"]
mask = [i for i, w in enumerate(da_words) if w in control_da]
torch.save({
    "words": [da_words[i] for i in mask],
    "matrix": da_matrix[mask]
}, "biggraph_control_da122.pt")
print(f"DA control BigGraph: {len(mask)} words")