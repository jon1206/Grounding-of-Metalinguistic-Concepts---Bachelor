from wordfreq import top_n_list

# Get top 20000 Danish words
danish_words = top_n_list('da', 20000)

print(f"Total Danish words: {len(danish_words)}")
print(f"First 30: {danish_words[:30]}")

with open("danish_20k.txt", "w", encoding="utf-8") as f:
    for w in danish_words:
        f.write(w + "\n")

print("Saved danish_20k.txt")