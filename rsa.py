"""
Representational Similarity Analysis (RSA).
Compares the structure of LLM and knowledge graph embedding spaces
without any alignment. Simply compares pairwise distance matrices.

Usage:
    python rsa.py --lm gpt2_small_meta.pt --ref biggraph_metalinguistic_en.pt
"""

import argparse
import json
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def load_embeddings(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    words = data["words"]
    matrix = data["matrix"].numpy().astype(np.float64)
    return words, matrix


def filter_to_shared_vocab(lm_words, lm_matrix, ref_words, ref_matrix):
    ref_word_to_idx = {w: i for i, w in enumerate(ref_words)}
    shared_words = []
    lm_indices = []
    ref_indices = []
    for i, w in enumerate(lm_words):
        w_lower = w.lower()
        if w_lower in ref_word_to_idx:
            shared_words.append(w_lower)
            lm_indices.append(i)
            ref_indices.append(ref_word_to_idx[w_lower])
    return shared_words, lm_matrix[lm_indices], ref_matrix[ref_indices]


def compute_rsa(lm_matrix, ref_matrix, metric="euclidean"):
    """
    Compute RSA score between two embedding matrices.
    1. Compute pairwise distance matrix (RDM) for each
    2. Flatten and compare with cosine similarity and Spearman correlation
    """
    # Compute representational dissimilarity matrices
    rdm_lm = pdist(lm_matrix, metric=metric)
    rdm_ref = pdist(ref_matrix, metric=metric)

    # Cosine similarity between flattened RDMs (as in the paper)
    cos_sim = np.dot(rdm_lm, rdm_ref) / (np.linalg.norm(rdm_lm) * np.linalg.norm(rdm_ref))

    # Spearman correlation (common alternative)
    spearman_r, spearman_p = spearmanr(rdm_lm, rdm_ref)

    return cos_sim, spearman_r, spearman_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load
    print(f"Loading LM: {args.lm}")
    lm_words, lm_matrix = load_embeddings(args.lm)
    print(f"  Shape: {lm_matrix.shape}")

    print(f"Loading ref: {args.ref}")
    ref_words, ref_matrix = load_embeddings(args.ref)
    print(f"  Shape: {ref_matrix.shape}")

    # Filter
    shared_words, lm_filtered, ref_filtered = filter_to_shared_vocab(
        lm_words, lm_matrix, ref_words, ref_matrix
    )
    print(f"Shared vocabulary: {len(shared_words)} words")
    print(f"Number of pairwise distances: {len(shared_words) * (len(shared_words) - 1) // 2}")

    # Compute RSA with euclidean distance
    print("\nRSA (euclidean):")
    cos_euc, spear_euc, p_euc = compute_rsa(lm_filtered, ref_filtered, metric="euclidean")
    print(f"  Cosine similarity: {cos_euc:.4f}")
    print(f"  Spearman r: {spear_euc:.4f} (p={p_euc:.2e})")

    # Compute RSA with cosine distance
    print("\nRSA (cosine):")
    cos_cos, spear_cos, p_cos = compute_rsa(lm_filtered, ref_filtered, metric="cosine")
    print(f"  Cosine similarity: {cos_cos:.4f}")
    print(f"  Spearman r: {spear_cos:.4f} (p={p_cos:.2e})")

    # Save
    if args.output:
        output = {
            "lm": args.lm,
            "ref": args.ref,
            "shared_vocab": len(shared_words),
            "euclidean": {
                "cosine_similarity": float(cos_euc),
                "spearman_r": float(spear_euc),
                "spearman_p": float(p_euc),
            },
            "cosine": {
                "cosine_similarity": float(cos_cos),
                "spearman_r": float(spear_cos),
                "spearman_p": float(p_cos),
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()