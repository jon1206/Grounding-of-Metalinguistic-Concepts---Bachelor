import requests
import json
import time

headers = {"User-Agent": "Bachelor-thesis-research/1.0"}
endpoint = "https://query.wikidata.org/sparql"

# Load the QIDs we already have
with open("metalinguistic_all_qids.json", "r") as f:
    en_terms = json.load(f)

print(f"English terms with QIDs: {len(en_terms)}")

# For each QID, get the Danish label
qid_to_da = {}
qids = list(en_terms.values())

for i in range(0, len(qids), 50):
    chunk = qids[i:i+50]
    values = " ".join(f"wd:{qid}" for qid in chunk)
    
    query = f"""
    SELECT ?item ?label WHERE {{
      VALUES ?item {{ {values} }}
      ?item rdfs:label ?label .
      FILTER(LANG(?label) = "da")
    }}
    """
    
    try:
        resp = requests.get(endpoint, params={"query": query, "format": "json"},
                          headers=headers, timeout=120)
        if resp.status_code == 200:
            for result in resp.json()["results"]["bindings"]:
                qid = result["item"]["value"].split("/")[-1]
                label = result["label"]["value"].lower()
                qid_to_da[qid] = label
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(2)

# Build mapping: english term -> danish term (via QID)
en_to_da = {}
for en_term, qid in en_terms.items():
    if qid in qid_to_da:
        en_to_da[en_term] = qid_to_da[qid]

print(f"Terms with Danish translations: {len(en_to_da)}")
print(f"Missing Danish: {sorted(set(en_terms.keys()) - set(en_to_da.keys()))}")

# Save
with open("metalinguistic_en_to_da.json", "w", encoding="utf-8") as f:
    json.dump(en_to_da, f, ensure_ascii=False, indent=2)

# Save Danish word list
da_terms = sorted(set(en_to_da.values()))
with open("metalinguistic_da.txt", "w", encoding="utf-8") as f:
    for t in da_terms:
        f.write(t + "\n")

print(f"\nUnique Danish terms: {len(da_terms)}")
print("Sample translations:")
for en, da in list(en_to_da.items())[:20]:
    print(f"  {en} -> {da}")