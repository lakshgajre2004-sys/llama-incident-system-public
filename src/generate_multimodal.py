# List repository structure and show key files to inspect
import os, textwrap, json, glob
root = "/mnt/data/Resaerch_Paper_extracted/llama-incident-system"
for path, dirs, files in os.walk(root):
    # limit depth
    if path.count(os.sep) - root.count(os.sep) > 2:
        continue
    print("DIR:", path)
    for f in files:
        print("  ", f)
print("\nNow showing src/generate_data.py and src/train.py content (if present):\n")
for fname in ["src/generate_data.py","src/train.py","src/utils.py","src/evaluate.py"]:
    p = os.path.join(root, fname)
    if os.path.exists(p):
        print("=== FILE:", fname, "===\n")
        print(open(p, "r", encoding="utf-8", errors="replace").read())
    else:
        print("=== FILE:", fname, "MISSING ===\n")
