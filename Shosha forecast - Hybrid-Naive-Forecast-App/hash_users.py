# hash_users.py
import json
import hashlib

IN_FILE = "users.json"        # your current file with plaintext passwords
OUT_FILE = "users_hashed.json"  # output file with hashed passwords

with open(IN_FILE, "r", encoding="utf-8") as f:
    users = json.load(f)

for u in users:
    pw = u.get("password", "")
    if pw:
        u["password"] = hashlib.sha256(pw.encode()).hexdigest()
    else:
        u["password"] = ""

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(users, f, indent=2)

print(f"Saved hashed users to {OUT_FILE}")
