#!/usr/bin/env python3
"""
Reset seeded data for a single user, keeping the user profile and goals.

Env:
  MONGO_URI   (required)
  DB_NAME     (default: Focus_DB)
  USER_ID     (default: prashanth)
  DRY_RUN     (default: true)  -> set to "false" to actually delete
"""
import os
from datetime import datetime, timezone
from pymongo import MongoClient
import certifi

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://prashanth01071995:pradsml%402025@cluster0.fsbic.mongodb.net/")
DB_NAME = os.environ.get("DB_NAME", "Focus_DB")
USER_ID   = os.getenv("USER_ID", "prashanth")
DRY_RUN   = (os.getenv("DRY_RUN", "true").lower() != "false")

if not MONGO_URI:
    raise SystemExit("MONGO_URI is required")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
client.admin.command("ping")
db = client[DB_NAME]

collections = ["sessions", "weekly_plans", "daily_targets", "reflections"]

print(f"[cfg] DB={DB_NAME} USER={USER_ID} DRY_RUN={DRY_RUN}")
print("Collections:", ", ".join(sorted(db.list_collection_names())))

def count_all():
    return {c: db[c].count_documents({"user": USER_ID}) for c in collections}

before = count_all()
print("\n[before] per-collection user-doc counts")
for c, n in before.items():
    print(f"  {c:14} : {n}")

if DRY_RUN:
    print("\n[dry-run] No deletes performed. Set DRY_RUN=false to apply.")
else:
    total_deleted = 0
    for c in collections:
        res = db[c].delete_many({"user": USER_ID})
        print(f"[deleted] {c:14} : {res.deleted_count}")
        total_deleted += res.deleted_count
    # Optional: also clear any old logs/user_days if they exist in your DB
    for maybe in ["logs", "user_days", "daily_rhythm"]:
        if maybe in db.list_collection_names():
            res = db[maybe].delete_many({"user": USER_ID})
            if res.deleted_count:
                print(f"[deleted] {maybe:14} : {res.deleted_count}")

    print(f"\n[done] Total deleted: {total_deleted} docs @ {datetime.now(timezone.utc).isoformat()}Z")

after = count_all()
print("\n[after] per-collection user-doc counts")
for c, n in after.items():
    print(f"  {c:14} : {n}")
