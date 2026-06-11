"""
model_name (line 12) and words (line 20) is modified for each run.
"""

import torch
from transformers import GPT2Model, AutoTokenizer

def load_wordlist(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

model_name = "KennethTM/gpt2-small-danish"
print(f"Loading {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

words = load_wordlist("da_122.txt")
print(f"Words: {len(words)}")

embeddings = {}
with torch.no_grad():
    for i, word in enumerate(words):
        try:
            text = f"{word} er"
            ids = tokenizer.encode(text, add_special_tokens=False)
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            
            input_ids = torch.tensor([ids]).to(device)
            output = model(input_ids).last_hidden_state.squeeze(0)
            
            word_end = len(word_ids)
            embeddings[word] = output[:word_end].mean(dim=0).cpu()
        except:
            pass
        
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(words)}")

print(f"Extracted: {len(embeddings)}")
word_order = list(embeddings.keys())
matrix = torch.stack([embeddings[w] for w in word_order])
torch.save({"words": word_order, "matrix": matrix}, "gpt2_small_da122.pt")
print(f"Saved - shape: {matrix.shape}")