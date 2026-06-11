"""
Extract word embeddings from BERT/DistilBERT/XLM-R using a template sentence.

Usage:
    python extract_encoders.py --model bert-base-multilingual-cased --wordlist 20k.txt --output mbert_en.pt
    python extract_encoders.py --model distilbert-base-multilingual-cased --wordlist 20k.txt --output distilmbert_en.pt
    python extract_encoders.py --model xlm-roberta-base --wordlist 20k.txt --output xlmr_base_en.pt
    python extract_encoders.py --model xlm-roberta-large --wordlist danish_20k.txt --output xlmr_large_da.pt
"""

import argparse
import torch
from transformers import AutoModel, AutoTokenizer


def load_wordlist(path):
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def extract_template_embeddings(model, tokenizer, words, template="The {} is"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = {}
    skipped = []

    with torch.no_grad():
        for i, word in enumerate(words):
            try:
                text = template.format(word)
                encoded = tokenizer(text, return_tensors="pt").to(device)
                output = model(**encoded).last_hidden_state.squeeze(0)

                # Tokenize template parts to find word position
                # Tokenize the prefix (e.g., "The ") to find where word starts
                prefix = template.split("{}")[0]
                prefix_tokens = tokenizer(prefix, add_special_tokens=True)["input_ids"]
                # Remove the [SEP]/</s> token at end of prefix
                word_start = len(prefix_tokens) - 1

                # Tokenize just the word to find how many tokens it uses
                word_tokens = tokenizer.tokenize(word)
                word_end = word_start + len(word_tokens)

                if word_end > output.shape[0]:
                    word_end = output.shape[0] - 1

                if word_start < word_end:
                    word_embedding = output[word_start:word_end].mean(dim=0).cpu()
                    embeddings[word] = word_embedding
                else:
                    skipped.append(word)

            except Exception as e:
                skipped.append(word)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(words)}")

    if skipped:
        print(f"Skipped {len(skipped)} words: {skipped[:10]}...")

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--wordlist", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--template", type=str, default="The {} is",
                        help="Template sentence with {} as word placeholder")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    print(f"Loading word list from: {args.wordlist}")
    words = load_wordlist(args.wordlist)
    print(f"  Found {len(words)} words")
    print(f"  Template: '{args.template}'")

    print("Extracting template embeddings...")
    embeddings = extract_template_embeddings(model, tokenizer, words, template=args.template)

    print(f"  Extracted embeddings for {len(embeddings)} words")
    if embeddings:
        sample_word = list(embeddings.keys())[0]
        print(f"  Embedding dimension: {embeddings[sample_word].shape[0]}")

    word_order = list(embeddings.keys())
    matrix = torch.stack([embeddings[w] for w in word_order])

    torch.save({"words": word_order, "matrix": matrix}, args.output)
    print(f"Saved {args.output} - matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()