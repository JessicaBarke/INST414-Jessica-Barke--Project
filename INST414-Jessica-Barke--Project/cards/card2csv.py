import json, csv, os

pairs = [
    ("cards/addiction_card/dataset-metadata.json", "cards/addiction_card.csv"),
    ("cards/productivity_card/dataset-metadata.json", "cards/productivity_card.csv"),
]
fields = ["title","id","subtitle","description","isPrivate"]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    obj = json.loads(txt)
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} parsed to {type(obj).__name__}, expected dict")
    return obj

def flat(x):
    return "" if x is None else str(x).replace("\r"," ").replace("\n"," ")

for src, dst in pairs:
    if not os.path.exists(src):
        print("Missing:", src)
        continue
    meta = load(src)
    row = {k: flat(meta.get(k, "")) for k in fields}
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerow(row)
    print("Wrote", dst)
