# --- top of page (pages/focus.py) ---
import os
from datetime import datetime, timezone
import certifi
import pytz
import streamlit as st
from pymongo import MongoClient

IST = pytz.timezone("Asia/Kolkata")

def today_iso():
    return datetime.now(IST).date().isoformat()

@st.cache_resource
def _db():
    uri = st.secrets.get("mongo_url") or os.getenv("mongo_uri")
    dbname = st.secrets.get("DB_NAME") or os.getenv("DB_NAME", "Focus_DB")
    if not uri:
        st.error("MONGO_URI is not configured (set in .streamlit/secrets.toml).")
        st.stop()
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=6000,
            tlsCAFile=certifi.where()  # ensure Atlas CA works
        )
        client.admin.command("ping")  # fast fail if blocked/invalid
        return client[dbname]
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop()

db = _db()
_user = st.secrets.get("USER_ID") or os.getenv("USER_ID", "prashanth")

# ---- replace your old query with this: ----
today = today_iso()
tgt_doc = db.daily_targets.find_one({"user": _user, "date_ist": today})
# tgt_doc may be None on first run — handle gracefully:
target_pomos = (tgt_doc or {}).get("target_pomos")
