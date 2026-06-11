metalinguistic_terms = [
    # Syntactic classes / Parts of speech
    "noun", "verb", "adjective", "adverb", "pronoun", "preposition",
    "conjunction", "interjection", "determiner", "participle",
    "infinitive", "gerund", "copula",
    
    # Morphological concepts (unambiguous)
    "morpheme", "prefix", "suffix", "affix", "inflection",
    "declension", "conjugation", "lemma", "derivation",
    "diminutive", "augmentative", "pluralization",
    
    # Phonological concepts (unambiguous)
    "phoneme", "vowel", "consonant", "syllable", "diphthong",
    "fricative", "plosive", "phonology", "phonetics",
    "allophone", "monophthong", "triphthong",
    
    # Syntactic concepts (unambiguous)
    "syntax", "clause", "predicate", "subordination",
    "complementizer", "relativizer",
    
    # Semantic concepts (unambiguous)
    "synonym", "antonym", "homonym", "polysemy", "homophone",
    "hypernym", "hyponym", "meronym",
    
    # Grammatical cases (unambiguous Latin-derived terms)
    "nominative", "accusative", "dative", "genitive",
    "ablative", "vocative", "instrumental", "locative",
    
    # Grammatical moods (unambiguous)
    "indicative", "subjunctive", "imperative", "conditional",
    "optative", "jussive",
    
    # Aspect/tense (unambiguous)
    "perfective", "imperfective", "aorist", "pluperfect",
    "preterite",
    
    # Typological features
    "agglutinative", "isolating", "fusional", "polysynthetic",
    "ergative", "accusative",  # already there but typological too
    
    # Language names (these are unambiguous)
    "english", "danish", "german", "french", "spanish", "italian",
    "portuguese", "dutch", "swedish", "norwegian", "icelandic",
    "finnish", "russian", "polish", "czech", "greek", "latin",
    "arabic", "chinese", "japanese", "korean", "hindi", "turkish",
    "hungarian", "romanian", "bulgarian", "croatian", "serbian",
    "hebrew", "persian", "thai", "vietnamese", "indonesian",
    "swahili", "basque", "catalan", "galician", "esperanto",
    
    # Writing systems (unambiguous)
    "alphabet", "syllabary", "logography", "orthography",
    "grapheme", "diacritic", "ligature",
    
    # Linguistics subfields
    "phonology", "morphology", "syntax", "semantics", "pragmatics",
    "etymology", "lexicography", "sociolinguistics", "psycholinguistics",
    "neurolinguistics", "typology",
    
    # Other unambiguous linguistic terms
    "dialect", "idiolect", "sociolect", "pidgin", "creole",
    "lingua franca", "cognate", "calque", "neologism",
    "collocation", "idiom", "etymology",
    "bilingualism", "multilingualism", "diglossia",
    "borrowing", "loanword",
]

# Remove duplicates and multi-word terms (for BigGraph matching)
single_word = sorted(set(t for t in metalinguistic_terms if " " not in t))

print(f"Single-word terms: {len(single_word)}")

with open("metalinguistic_en.txt", "w", encoding="utf-8") as f:
    for term in single_word:
        f.write(term + "\n")

print("Saved metalinguistic_en.txt")
for t in single_word:
    print(f"  {t}")