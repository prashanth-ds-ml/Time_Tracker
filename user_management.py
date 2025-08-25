import streamlit as st
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional
import pandas as pd
import pytz, hashlib, math
from pymongo import MongoClient

# ====== GLOBALS ======
POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone("Asia/Kolkata")
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# ====== DB INIT ======
@st.cache_resource
def init_database():
    try:
        MONGO_URI = st.secrets["mongo_uri"]
        client = MongoClient(MONGO_URI)
        return client["time_tracker_db"]
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

db = init_database()
collection_logs = db["logs"]                 # sessions / targets / notes
users_collection = db["users"]               # user settings
collection_goals = db["goals"]               # weekly goals catalog
collection_plans = db["weekly_plans"]        # plan per week
collection_reflections = db["reflections"]   # daily reflections

# ====== TIME HELPERS ======
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)
    return start, end

def week_day_counts(week_start: date) -> Tuple[int, int]:
    wd = sum((week_start + timedelta(days=i)).weekday() < 5 for i in range(7))
    we = 7 - wd
    return wd, we

# ====== MATH HELPERS ======
def safe_div(n, d, default=0.0):
    try:
        if d in (0, None): return default
        return float(n) / float(d)
    except Exception:
        return default

def pct_or_dash(n, d, decimals=0):
    if not d: return "—"
    return f"{100.0*safe_div(n, d):.{decimals}f}%"

def gini_from_counts(counts):
    arr = [c for c in counts if c is not None and c >= 0]
    if not arr: return 0.0
    arr = sorted(arr); n = len(arr); s = sum(arr)
    if s == 0: return 0.0
    cum = sum((i+1)*x for i, x in enumerate(arr))
    return (2.0*cum)/(n*s) - (n+1.0)/n

def entropy_norm_from_counts(counts):
    arr = [c for c in counts if c and c > 0]
    k = len(arr)
    if k <= 1: return 0.0
    s = float(sum(arr))
    import math
    H = -sum((c/s)*math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

def time_to_minutes(tstr: Optional[str]) -> Optional[int]:
    if not tstr or not isinstance(tstr, str): return None
    try:
        dt = datetime.strptime(tstr, "%I:%M %p")
        return dt.hour*60 + dt.minute
    except Exception:
        return None

# ====== SOUND ======
def sound_alert():
    st.components.v1.html(
        f"""
        <audio id="beep" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
        <script>
            const a = document.getElementById('beep');
            if (a) {{ a.volume = 0.6; a.play().catch(()=>{{}}); }}
        </script>
        """,
        height=0,
    )

# ====== USERS ======
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

@st.cache_data(ttl=120)
def get_user_settings(username: str) -> Dict:
    doc = users_collection.find_one({"username": username})
    if not doc:
        add_user(username)
        doc = users_collection.find_one({"username": username})
    return {
        "weekday_poms": int(doc.get("weekday_poms", 3)),
        "weekend_poms": int(doc.get("weekend_poms", 5)),
        "auto_break": bool(doc.get("auto_break", True)),
        "custom_categories": list(doc.get("custom_categories", ["Learning","Projects","Research","Planning"])),
    }

# ====== DATA ACCESS ======
@st.cache_data(ttl=300)
def get_user_sessions(username: str) -> pd.DataFrame:
    recs = list(collection_logs.find({"type": "Pomodoro", "user": username}))
    base_cols = ["type","date","time","pomodoro_type","duration","user","goal_id","task","category"]
    if not recs:
        return pd.DataFrame(columns=base_cols)
    df = pd.DataFrame(recs)
    for c in base_cols:
        if c not in df.columns:
            df[c] = "" if c in ("time","task","category") else None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    if "pomodoro_type" not in df.columns: df["pomodoro_type"] = "Work"
    if "goal_id" not in df.columns: df["goal_id"] = None
    if "category" not in df.columns: df["category"] = ""
    return df[base_cols]

def export_sessions_csv(user: str):
    df = get_user_sessions(user)
    if df.empty:
        st.info("No sessions to export.")
        return
    out = df.sort_values("date")
    st.download_button("⬇️ Export Sessions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"{user}_sessions.csv",
        mime="text/csv"
    )

# ====== GOALS ======
def upsert_goal(username: str, title: str, priority_weight: int, goal_type: str,
                status: str = "New", target_poms: int = 0) -> str:
    gid = hashlib.sha256(f"{username}|{title}".encode()).hexdigest()[:16]
    set_on_insert = {
        "_id": gid, "user": username, "title": title,
        "target_poms": int(target_poms), "poms_completed": 0,
        "created_at": datetime.utcnow(),
    }
    set_always = {
        "priority_weight": int(priority_weight), "goal_type": goal_type,
        "status": status, "updated_at": datetime.utcnow(),
    }
    collection_goals.update_one({"_id": gid}, {"$setOnInsert": set_on_insert, "$set": set_always}, upsert=True)
    return gid

@st.cache_data(ttl=60)
def fetch_goals(username: str, statuses: Optional[List[str]] = None) -> pd.DataFrame:
    q = {"user": username}
    if statuses: q["status"] = {"$in": statuses}
    recs = list(collection_goals.find(q))
    cols = ["_id","user","title","priority_weight","goal_type","status","target_poms","poms_completed","updated_at","created_at"]
    return pd.DataFrame(recs, columns=[c for c in cols if recs]) if recs else pd.DataFrame(columns=cols)

def goal_title_map(user: str) -> Dict[str, str]:
    docs = list(collection_goals.find({"user": user}, {"_id": 1, "title": 1}))
    return {d["_id"]: d["title"] for d in docs}

# ====== WEEKLY PLAN ======
def compute_weekly_capacity(settings: Dict, weekdays: int = 5, weekend_days: int = 2) -> int:
    return settings["weekday_poms"] * weekdays + settings["weekend_poms"] * weekend_days

def proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    total_w = max(1, sum(max(1, int(w)) for w in weights.values()))
    raw = {gid: (max(1, int(w)) / total_w) * total for gid, w in weights.items()}
    allocated = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(allocated.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid] - int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        idx = 0
        while diff != 0 and fracs:
            gid = fracs[idx % len(fracs)][0]
            allocated[gid] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1
    return allocated

def get_or_create_weekly_plan(username: str, d: Optional[date] = None) -> Dict:
    if d is None: d = now_ist().date()
    week_start, week_end = week_bounds_ist(d)
    pid = f"{username}|{week_start.isoformat()}"
    plan = collection_plans.find_one({"_id": pid})
    if plan: return plan
    settings = get_user_settings(username)
    wd, we = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(settings, weekdays=wd, weekend_days=we)
    doc = {
        "_id": pid, "user": username,
        "week_start": week_start.isoformat(), "week_end": week_end.isoformat(),
        "total_poms": total_poms, "goals": [], "allocations": {},
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    collection_plans.insert_one(doc)
    return doc

def save_plan_allocations(plan_id: str, goals: List[str], allocations: Dict[str, int]):
    goals_unique = sorted(set(goals))
    clean_alloc = {gid: int(max(0, int(allocations.get(gid, 0)))) for gid in goals_unique}
    collection_plans.update_one({"_id": plan_id},
        {"$set": {"goals": goals_unique, "allocations": clean_alloc, "updated_at": datetime.utcnow()}}
    )

# ====== BALANCE LOCK ======
def is_within_lock_window(plan: Dict, days_window: int = 3) -> bool:
    start = datetime.fromisoformat(plan["week_start"]).date()
    return (now_ist().date() - start).days <= (days_window - 1)

@st.cache_data(ttl=30)
def locked_goals_for_user_plan(username: str, plan: Dict, threshold_pct: float = 0.7, min_other: int = 3) -> List[str]:
    if not is_within_lock_window(plan): return []
    start = datetime.fromisoformat(plan["week_start"]).date()
    today = now_ist().date()
    df = get_user_sessions(username)
    if df.empty: return []
    mask_week = (df["date"].dt.date >= start) & (df["date"].dt.date <= today)
    dfw = df[mask_week & (df["pomodoro_type"] == "Work")].copy()
    if dfw.empty: return []
    if "goal_id" not in dfw.columns: dfw["goal_id"] = None
    by_goal = dfw.groupby(dfw["goal_id"].astype("object").fillna("NONE")).size().sort_values(ascending=False)
    total = int(by_goal.sum())
    if total < 4: return []
    top2 = by_goal.head(2).sum()
    if safe_div(top2, total) >= threshold_pct:
        dominating = [g for g in by_goal.head(2).index if g != "NONE"]
        others = by_goal[~by_goal.index.isin(dominating)]
        if len(others) == 0 or any(others < min_other): return dominating
    return []

# ====== SESSIONS ======
def save_pomodoro_session(user: str, is_break: bool, duration: int, goal_id: Optional[str], task: str, category_label: str):
    now = now_ist()
    doc = {
        "type": "Pomodoro",
        "date": now.date().isoformat(),
        "time": now.strftime("%I:%M %p"),
        "pomodoro_type": "Break" if is_break else "Work",
        "duration": int(duration),
        "user": user,
        "goal_id": goal_id if not is_break else None,
        "task": task if not is_break else "",
        "category": category_label if (category_label and not is_break) else "",
        "created_at": datetime.utcnow(),
    }
    collection_logs.insert_one(doc)
    if (not is_break) and goal_id:
        collection_goals.update_one({"_id": goal_id}, {"$inc": {"poms_completed": 1}, "$set": {"updated_at": datetime.utcnow()}})
    get_user_sessions.clear()

# ====== DAILY TARGETS ======
def get_adaptive_goal(active_days: int):
    if active_days <= 5:   return 1, "🌱 Building", "Start small - consistency over intensity"
    if active_days <= 12:  return 2, "🔥 Growing", "Building momentum - you're doing great!"
    if active_days <= 19:  return 3, "💪 Strong", "Push your limits - you're in the zone!"
    return 4, "🚀 Peak", "Excellence mode - maintain this peak!"

def save_daily_target(target:int, user:str):
    today = now_ist().date().isoformat()
    doc = {"type":"DailyTarget","date": today, "target": int(target), "user": user, "created_at": datetime.utcnow()}
    collection_logs.update_one({"type":"DailyTarget","date": today,"user": user},{"$set": doc}, upsert=True)

def get_daily_target(user:str):
    today = now_ist().date().isoformat()
    doc = collection_logs.find_one({"type":"DailyTarget","date": today,"user": user})
    return int(doc["target"]) if doc else None
