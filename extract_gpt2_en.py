"""
Extract word embeddings from GPT-2 using a template sentence for context.
Uses "A [word] is" template and extracts the word's hidden state.

Usage:
    python extract_gpt2_en.py --model gpt2 --wordlist 20k.txt --output gpt2_small_tmpl.pt
"""

import argparse
import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer


def load_wordlist(path):
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def extract_template_embeddings(model, tokenizer, words, batch_size=1):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = {}
    skipped = []

    with torch.no_grad():
        for i, word in enumerate(words):
            try:
                template = f"A {word} is"
                ids = tokenizer.encode(template, add_special_tokens=False)
                word_ids = tokenizer.encode(word, add_special_tokens=False)

                input_ids = torch.tensor([ids]).to(device)
                output = model(input_ids).last_hidden_state.squeeze(0)

                # "A" is token 0, word starts at token 1
                word_start = 1
                word_end = 1 + len(word_ids)
                word_embedding = output[word_start:word_end].mean(dim=0).cpu()
                embeddings[word] = word_embedding

            except Exception as e:
                skipped.append(word)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(words)}")

    if skipped:
        print(f"Skipped {len(skipped)} words: {skipped[:10]}...")

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--wordlist", type=str, required=True)
    parser.add_argument("--output", type=str, default="embeddings_tmpl.pt")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model)

    print(f"Loading word list from: {args.wordlist}")
    words = load_wordlist(args.wordlist)
    print(f"  Found {len(words)} words")

    print("Extracting template embeddings...")
    embeddings = extract_template_embeddings(model, tokenizer, words)

    print(f"  Extracted embeddings for {len(embeddings)} words")
    sample_word = list(embeddings.keys())[0]
    print(f"  Embedding dimension: {embeddings[sample_word].shape[0]}")

    word_order = list(embeddings.keys())
    matrix = torch.stack([embeddings[w] for w in word_order])

    torch.save({"words": word_order, "matrix": matrix}, args.output)
    print(f"Saved {args.output} - matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()