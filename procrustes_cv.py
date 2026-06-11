"""
Procrustes alignment with k-fold cross-validation.
Use this for small datasets where a single 80/20 split is unreliable.

Usage:
    python procrustes_cv.py --lm gpt2_small_meta.pt --ref biggraph_metalinguistic_en.pt --output results.json --folds 5
"""

import argparse
import json
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist


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


def align_and_evaluate(lm_train, ref_train, lm_test, ref_test):
    """Run Procrustes on one fold and return P@k results."""
    # Center
    lm_mean = lm_train.mean(axis=0)
    ref_mean = ref_train.mean(axis=0)
    lm_train_c = lm_train - lm_mean
    lm_test_c = lm_test - lm_mean
    ref_train_c = ref_train - ref_mean
    ref_test_c = ref_test - ref_mean

    # PCA reduce LM
    max_components = min(lm_train_c.shape[0] - 1, lm_train_c.shape[1], ref_train_c.shape[1])
    pca = PCA(n_components=max_components)
    lm_train_r = pca.fit_transform(lm_train_c)
    lm_test_r = pca.transform(lm_test_c)

    # If needed, reduce ref too
    if max_components < ref_train_c.shape[1]:
        pca_ref = PCA(n_components=max_components)
        ref_train_c = pca_ref.fit_transform(ref_train_c)
        ref_test_c = pca_ref.transform(ref_test_c)

    # Normalize
    lm_train_n = lm_train_r / np.linalg.norm(lm_train_r, axis=1, keepdims=True)
    lm_test_n = lm_test_r / np.linalg.norm(lm_test_r, axis=1, keepdims=True)
    ref_train_n = ref_train_c / np.linalg.norm(ref_train_c, axis=1, keepdims=True)
    ref_test_n = ref_test_c / np.linalg.norm(ref_test_c, axis=1, keepdims=True)

    # Procrustes
    R, _ = orthogonal_procrustes(lm_train_n, ref_train_n)
    lm_test_aligned = lm_test_n @ R

    # Evaluate
    dists = cdist(lm_test_aligned, ref_test_n, metric="cosine")
    n = len(dists)
    results = {}
    for k in [1, 5, 10, 20]:
        hits = sum(1 for i in range(n) if i in np.argsort(dists[i])[:k])
        results[k] = hits / n
    return results, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--output", type=str, default="results_cv.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
    n = len(shared_words)
    print(f"Shared vocabulary: {n} words")
    print(f"Running {args.folds}-fold cross-validation...")

    # K-fold CV
    np.random.seed(args.seed)
    indices = np.random.permutation(n)
    fold_size = n // args.folds

    all_results = {k: [] for k in [1, 5, 10, 20]}
    total_test = 0

    for fold in range(args.folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < args.folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        lm_train = lm_filtered[train_idx]
        lm_test = lm_filtered[test_idx]
        ref_train = ref_filtered[train_idx]
        ref_test = ref_filtered[test_idx]

        results, test_n = align_and_evaluate(lm_train, ref_train, lm_test, ref_test)

        for k in [1, 5, 10, 20]:
            all_results[k].append(results[k])

        total_test += test_n
        print(f"  Fold {fold+1}: P@1={results[1]:.4f}, P@5={results[5]:.4f}, "
              f"P@10={results[10]:.4f}, P@20={results[20]:.4f} (n={test_n})")

    # Average across folds
    print(f"\nAverage across {args.folds} folds:")
    avg_results = {}
    for k in [1, 5, 10, 20]:
        mean = np.mean(all_results[k])
        std = np.std(all_results[k])
        avg_results[f"P@{k}"] = {"mean": float(mean), "std": float(std)}
        print(f"  P@{k}: {mean:.4f} ± {std:.4f}")

    print(f"  Random baseline P@1: {1/fold_size:.4f}")

    # Save
    output = {
        "lm": args.lm,
        "ref": args.ref,
        "shared_vocab": n,
        "folds": args.folds,
        "fold_size": fold_size,
        "results": avg_results,
        "per_fold": {f"P@{k}": all_results[k] for k in [1, 5, 10, 20]},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()