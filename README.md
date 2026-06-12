# Grounding of Metalinguistic Concepts — Code and Results

Supplementary material for the bachelor thesis *Grounding of Metalinguistic
Concepts* (Jensen & Larsen, University of Copenhagen, June 2026). Contains
the pipeline scripts, vocabularies, QID mappings, reference embeddings, and
per-fold results behind all tables in the thesis.

## Pipeline overview

```
Vocabulary construction          Reference spaces (BigGraph)
----------------------          ---------------------------
build_meta_vocab_en.py     -->  build_meta_reference_en.py   (QID matching, §3.2)
translate_meta_vocab_da.py -->  build_meta_reference_da.py   (same QID vectors, Danish labels)
build_general_vocab_da.py  -->  build_general_reference_da.py (label matching)
                                build_control_references.py   (top-138 EN / top-122 DA)

LM embedding extraction (§4.2)         Evaluation (§4.5)
------------------------------         -----------------
extract_gpt2_en.py    ("A {} is")      procrustes_cv.py  -> Tables 2-3
extract_gpt2_da.py    ("{} er")        rsa.py            -> Tables 4-5
extract_encoders.py   ("The {} is" /
                       "En {} er")
```

## Scripts

- `build_meta_vocab_en.py` — curated candidate list of unambiguous English
  metalinguistic terms; the final 138-term vocabulary is the subset matched
  to BigGraph via QIDs.
- `build_meta_reference_en.py` — matches terms to BigGraph entities by
  Wikidata QID and saves `biggraph_metalinguistic_en.pt` and the final
  `metalinguistic_en.txt`.
- `translate_meta_vocab_da.py` — queries the Wikidata SPARQL endpoint for
  Danish labels of the English QIDs, producing the 122-term Danish
  vocabulary (`metalinguistic_da.txt`, `metalinguistic_en_to_da.json`).
- `build_meta_reference_da.py` — builds the Danish metalinguistic reference
  space. Note: it deliberately reuses the *same* QID vectors as English
  (identical concepts, Danish labels), so EN/DA comparisons hold the target
  space fixed and isolate language-model-side differences.
- `build_general_vocab_da.py` — top-20,000 Danish words via `wordfreq`.
- `build_general_reference_da.py` — matches the Danish list against
  BigGraph `@da` labels (single words; collisions resolved by first match).
- `build_general_reference_en.py` — **reconstruction** of the English
  label-matching script (the original was not preserved): identical logic
  to the Danish version, with `@en` labels and the English 20k list. The
  original script's output, `biggraph_embeddings.pt`, is included and can
  be used to verify the reconstruction (see the script's docstring).
- `build_control_references.py` — slices the size-matched control word
  lists (top-138 EN / top-122 DA high-frequency words) out of the BigGraph
  reference embeddings.
- `extract_gpt2_en.py`, `extract_gpt2_da.py`, `extract_encoders.py` —
  template-based contextual embedding extraction: final-layer hidden states,
  averaged over subword tokens. `extract_gpt2_da.py` was run per word list
  by editing the hardcoded model/wordlist/output names.
- `procrustes_cv.py` — Procrustes alignment with 5-fold CV (fixed seed 42;
  reported std is the population standard deviation over folds). Produces
  the numbers in **Tables 2 and 3**.
- `rsa.py` — Representational Similarity Analysis. Produces **Tables 4
  and 5**; the script computes both euclidean- and cosine-distance RDMs,
  and **the thesis reports the `cosine` block**.

## Data

- `Results/` — one JSON per (vocabulary, method, model, language) run,
  including per-fold P@k values. Every number in Tables 2–5 can be
  recomputed from these files.
- `metalinguistic_en.txt` / `metalinguistic_da.txt` — final vocabularies
  (138 / 122 terms; also printed in Appendix A of the thesis).
- `metalinguistic_qids.json`, `extra_metalinguistic_qids.json`,
  `metalinguistic_all_qids.json` — the manually verified term-to-QID
  mappings (§4.4); `metalinguistic_en_to_da.json` — the QID-mediated
  English-to-Danish translations.
- `en_138.txt`, `da_122.txt` — size-matched control word lists
  (top-138 / top-122 high-frequency words; the English list was
  constructed manually).
- `danish_20k.txt` — Danish general vocabulary. The English general
  vocabulary is the first20hours 20k list:
  https://github.com/first20hours/google-10000-english/blob/master/20k.txt
- `*.pt` reference embeddings — `biggraph_metalinguistic_en.pt`,
  `biggraph_metalinguistic_da.pt`, `biggraph_control_en138.pt`,
  `biggraph_control_da122.pt`, `biggraph_danish_embeddings.pt`,
  `biggraph_embeddings.pt` (English general).

## Reproducing a result

```bash
# English metalinguistic, GPT-2 large -> Table 2 (meta) / Table 5 (cosine)
python extract_gpt2_en.py --model gpt2-large \
    --wordlist metalinguistic_en.txt --output gpt2_large_meta.pt
python procrustes_cv.py --lm gpt2_large_meta.pt \
    --ref biggraph_metalinguistic_en.pt --output out_cv.json --folds 5
python rsa.py --lm gpt2_large_meta.pt \
    --ref biggraph_metalinguistic_en.pt --output out_rsa.json
```

## Not included

- **Large binaries:** the raw pretrained BigGraph WikiData dump
  (https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html;
  the reference-builder scripts contain machine-specific paths to it —
  edit before running) and the language models (Hugging Face: `gpt2`,
  `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `KennethTM/gpt2-small-danish`,
  `KennethTM/gpt2-medium-danish`, `bert-base-multilingual-cased`,
  `distilbert-base-multilingual-cased`, `xlm-roberta-base`,
  `xlm-roberta-large`).

## Requirements

```
torch
transformers
numpy
scipy
scikit-learn
wordfreq
requests
matplotlib
```

## Notes

- `torch.load(..., weights_only=False)` is used to read the bundled `.pt`
  files; only load `.pt` files from trusted sources.
- Random seed is fixed (42) in `procrustes_cv.py`, so fold assignment is
  deterministic for a given vocabulary.
