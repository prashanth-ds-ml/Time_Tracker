# user_management.py
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta, date
import pytz
import hashlib
from typing import List, Dict, Optional
import math

# --- Timezone ---
IST = pytz.timezone('Asia/Kolkata')

# --- DB INIT ---
@st.cache_resource
def init_database():
    try:
        MONGO_URI = st.secrets["mongo_uri"]
        client = MongoClient(MONGO_URI)
        db = client["time_tracker_db"]
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

db = init_database()
collection_logs = db["logs"]
users_collection = db["users"]
collection_goals = db["goals"]
collection_plans = db["weekly_plans"]
collection_reflections = db["reflections"]

# --- Time / Week helpers ---
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> (date, date):
    weekday = d.weekday()  # 0=Mon
    start = d - timedelta(days=weekday)
    end = start + timedelta(days=6)
    return start, end

def week_day_counts(week_start: date) -> (int, int):
    wd = 0; we = 0
    for i in range(7):
        day = week_start + timedelta(days=i)
        if day.weekday() < 5: wd += 1
        else: we += 1
    return wd, we

# --- Math helpers ---
def safe_div(n, d, default=0.0):
    try:
        if d is None or d == 0:
            return default
        return float(n) / float(d)
    except Exception:
        return default

def pct_or_dash(n, d, decimals=0):
    if d is None or d <= 0:
        return "—"
    pct = 100.0 * safe_div(n, d, default=0.0)
    return f"{pct:.{decimals}f}%"

def gini_from_counts(counts):
    arr = [c for c in counts if c is not None and c >= 0]
    if not arr: return 0.0
    arr = sorted(arr)
    n = len(arr); s = sum(arr)
    if s == 0: return 0.0
    cum = 0.0
    for i, x in enumerate(arr, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n

def entropy_norm_from_counts(counts):
    arr = [c for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1: return 0.0
    s = float(sum(arr))
    H = -sum((c/s) * math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

def time_to_minutes(tstr):
    from datetime import datetime as dt
    try:
        parsed = dt.strptime(tstr, "%I:%M %p")
        return parsed.hour*60 + parsed.minute
    except Exception:
        return None

# --- Data access / caches ---
@st.cache_data(ttl=300)
def get_user_sessions(username: str) -> pd.DataFrame:
    recs = list(collection_logs.find({"type": "Pomodoro", "user": username}))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    if 'goal_id' not in df.columns:
        df['goal_id'] = None
    if 'category' not in df.columns:
        df['category'] = ''
    if 'pomodoro_type' not in df.columns:
        df['pomodoro_type'] = 'Work'
    return df

@st.cache_data(ttl=120)
def get_user_settings(username: str) -> Dict:
    doc = users_collection.find_one({"username": username})
    if not doc:
        users_collection.insert_one({
            "username": username,
            "created_at": datetime.utcnow(),
            "weekday_poms": 3,
            "weekend_poms": 5,
            "auto_break": True,
            "custom_categories": ["Learning", "Projects", "Research", "Planning"],
        })
        doc = users_collection.find_one({"username": username})
    return {
        "weekday_poms": int(doc.get("weekday_poms", 3)),
        "weekend_poms": int(doc.get("weekend_poms", 5)),
        "auto_break": bool(doc.get("auto_break", True)),
        "custom_categories": list(doc.get("custom_categories", ["Learning","Projects","Research","Planning"]))
    }

@st.cache_data(ttl=60)
def get_all_users() -> List[str]:
    return [u["username"] for u in users_collection.find({}, {"_id": 0, "username": 1})]

def add_user(username: str) -> bool:
    if not users_collection.find_one({"username": username}):
        users_collection.insert_one({
            "username": username,
            "created_at": datetime.utcnow(),
            "weekday_poms": 3,
            "weekend_poms": 5,
            "auto_break": True,
            "custom_categories": ["Learning", "Projects", "Research", "Planning"],
        })
        get_all_users.clear()
        return True
    return False

def upsert_goal(username: str, title: str, priority_weight: int, goal_type: str, status: str = "New", target_poms: int = 0) -> str:
    gid = hashlib.sha256(f"{username}|{title}".encode()).hexdigest()[:16]
    set_on_insert = {
        "_id": gid, "user": username, "title": title,
        "target_poms": int(target_poms), "poms_completed": 0,
        "created_at": datetime.utcnow(),
    }
    set_always = {
        "priority_weight": int(priority_weight),
        "goal_type": goal_type,
        "status": status,
        "updated_at": datetime.utcnow(),
    }
    collection_goals.update_one({"_id": gid}, {"$setOnInsert": set_on_insert, "$set": set_always}, upsert=True)
    return gid

@st.cache_data(ttl=60)
def fetch_goals(username: str, statuses: Optional[List[str]] = None) -> pd.DataFrame:
    q = {"user": username}
    if statuses:
        q["status"] = {"$in": statuses}
    recs = list(collection_goals.find(q))
    return pd.DataFrame(recs) if recs else pd.DataFrame(
        columns=["_id","user","title","priority_weight","goal_type","status","target_poms","poms_completed"]
    )

def goal_title_map(user: str) -> Dict[str, str]:
    docs = list(collection_goals.find({"user": user}, {"_id": 1, "title": 1}))
    return {d["_id"]: d["title"] for d in docs}

# --- Admin / utils ---
def ensure_indexes():
    try:
        collection_logs.create_index([("user",1),("type",1),("date",1)], name="user_type_date")
        collection_logs.create_index([("goal_id",1)], name="goal_id")
        collection_goals.create_index([("user",1),("title",1)], unique=True, name="user_title_unique")
        collection_plans.create_index([("user",1),("week_start",1)], name="user_week")
        collection_reflections.create_index([("user",1),("date",1)], name="user_date")
        st.success("Indexes ensured/created.")
    except Exception as e:
        st.warning(f"Index creation notice: {e}")

def export_sessions_csv(user: str):
    df = get_user_sessions(user)
    if df.empty:
        st.info("No sessions to export.")
        return
    out = df.sort_values("date")
    st.download_button("⬇️ Export Sessions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"{user}_sessions.csv", mime="text/csv")
