# app.py
import os
import uuid
import time
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, Optional, List, Tuple

import certifi
import pytz
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from pymongo import MongoClient
from pymongo.errors import WriteError
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Config / constants
# ──────────────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
FINISH_SOUND_URL = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

CATEGORIES = ["Learning","Projects","Certification","Career","Health","Wellbeing","Start-up","Other"]
ALLOWED_ACTIVITY_TYPES = {"exercise", "meditation", "breathing", "other"}
ALLOWED_BUCKETS = {"current", "backlog"}

# ──────────────────────────────────────────────────────────────────────────────
# Small utils
# ──────────────────────────────────────────────────────────────────────────────
def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def play_finish_sound():
    if not st.session_state.get("sound_on", True):
        return
    st.markdown(
        f"""
        <audio autoplay>
          <source src="{FINISH_SOUND_URL}" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True,
    )

def now_ist() -> datetime:
    return datetime.now(IST)

def today_iso() -> str:
    return now_ist().date().isoformat()

def utc_from_ist(dt_ist: datetime) -> datetime:
    return dt_ist.astimezone(timezone.utc)

def _to_utc_naive(dt: datetime) -> datetime:
    """Return UTC-naive datetime for Mongo 'date' type."""
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

def to_ist_display(dt: Optional[datetime]) -> datetime:
    """Make any Mongo datetime (usually UTC-naive) safely IST-aware for display."""
    if not isinstance(dt, datetime):
        return now_ist()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)

def week_key_from_datestr(datestr: str) -> str:
    y, m, d = map(int, datestr.split("-"))
    dt = datetime(y, m, d)
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_key_from_date(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def monday_from_week_key(week_key: str) -> datetime:
    year, wk = map(int, week_key.split("-"))
    return IST.localize(datetime.fromisocalendar(year, wk, 1))

def prev_week_key(week_key: str) -> str:
    mon = monday_from_week_key(week_key)
    prev_mon = mon - timedelta(days=7)
    iso = prev_mon.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_dates_list(week_key: str) -> List[str]:
    mon = monday_from_week_key(week_key).date()
    return [(mon + timedelta(days=i)).isoformat() for i in range(7)]

def pom_equiv(minutes: int) -> float:
    return round(float(minutes) / 25.0, 2)

def pct(n, d) -> float:
    return (n / d * 100.0) if d else 0.0

def choose_last_n(label: str, available_count: int, default: int = 6, cap: int = 12, key: str = "lastn"):
    if available_count <= 0:
        st.caption(f"{label}: 0")
        return 0
    if available_count == 1:
        st.caption(f"{label}: 1")
        return 1
    maxv = min(cap, int(available_count))
    val = min(default, maxv)
    try:
        return st.slider(label, min_value=1, max_value=maxv, value=val, key=key)
    except StreamlitAPIException:
        st.warning("Slider fallback in use.")
        return st.number_input(label, min_value=1, max_value=maxv, value=val, step=1, key=f"{key}_num")

def should_render_heavy() -> bool:
    """Throttle heavy, multi-aggregation blocks while the timer is running."""
    running = bool(st.session_state.get("timer", {}).get("running"))
    if not running:
        return True
    now = time.time()
    last = st.session_state.get("last_heavy", 0.0)
    if now - last >= 10.0:
        st.session_state["last_heavy"] = now
        return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit + DB
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Focus Timer", page_icon="⏱️", layout="wide")

@st.cache_resource
def get_db():
    uri = (st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI") or os.getenv("mongo_uri") or "").strip()
    dbname = (st.secrets.get("DB_NAME") or os.getenv("DB_NAME") or "Focus_DB").strip()
    if not uri:
        st.error("MONGO_URI is not configured.")
        st.stop()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000, tlsCAFile=certifi.where())
        client.admin.command("ping")
        return client[dbname]
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop()

db = get_db()
USER_ID = (st.secrets.get("USER_ID") or os.getenv("USER_ID") or "vineel").strip()

# ──────────────────────────────────────────────────────────────────────────────
# Cached reads / aggregations (fast!)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5, show_spinner=False)
def get_user(uid: str) -> Optional[Dict[str, Any]]:
    return db.users.find_one({"_id": uid})

@st.cache_data(ttl=5, show_spinner=False)
def get_goals(uid: str) -> List[Dict[str, Any]]:
    return list(db.goals.find({"user": uid}).sort("updated_at", -1))

@st.cache_data(ttl=5, show_spinner=False)
def get_goals_map(uid: str) -> Dict[str, Dict[str, Any]]:
    return {g["_id"]: g for g in get_goals(uid)}

@st.cache_data(ttl=5, show_spinner=False)
def get_week_plan(uid: str, week_key: str) -> Optional[Dict[str, Any]]:
    return db.weekly_plans.find_one({"user": uid, "week_key": week_key})

@st.cache_data(ttl=5, show_spinner=False)
def total_day_pe(uid: str, date_ist: str) -> float:
    pipeline = [
        {"$match": {"user": uid, "date_ist": date_ist, "t": "W"}},
        {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
    ]
    doc = next(iter(db.sessions.aggregate(pipeline)), None)
    return float(doc["pe"]) if doc else 0.0

@st.cache_data(ttl=5, show_spinner=False)
def aggregate_pe_by_goal_bucket(uid: str, week_key: str) -> Dict[str, Dict[str, float]]:
    """
    Returns: { goal_id: {"current": x, "backlog": y} }
    Skips sessions without goal_id (e.g., activity).
    """
    out: Dict[str, Dict[str, float]] = {}
    pipeline = [
        {"$match": {
            "user": uid,
            "week_key": week_key,
            "t": "W",
            "goal_id": {"$exists": True, "$ne": None}
        }},
        {"$group": {
            "_id": {"goal_id": "$goal_id", "alloc_bucket": "$alloc_bucket"},
            "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}
        }}
    ]
    for row in db.sessions.aggregate(pipeline):
        gid = row["_id"]["goal_id"]
        bucket = row["_id"]["alloc_bucket"] or "current"
        pe = float(row["pe"])
        if gid not in out:
            out[gid] = {"current": 0.0, "backlog": 0.0}
        if bucket not in out[gid]:
            out[gid][bucket] = 0.0
        out[gid][bucket] += pe
    return out

@st.cache_data(ttl=5, show_spinner=False)
def list_today_sessions(uid: str, date_ist: str) -> List[Dict[str, Any]]:
    return list(db.sessions.find({"user": uid, "date_ist": date_ist}).sort("started_at_ist", 1))

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (writes bust caches)
# ──────────────────────────────────────────────────────────────────────────────
def get_user_capacity_defaults(uid: str) -> Tuple[int, int]:
    u = get_user(uid) or {}
    prefs = (u.get("prefs") or {})
    wkday_default = int(prefs.get("weekday_poms", 3))
    wkend_default = int(prefs.get("weekend_poms", 6))
    return wkday_default, wkend_default

def get_rank_weight_map(uid: str) -> Dict[str, int]:
    u = get_user(uid) or {}
    rwm = ((u.get("prefs") or {}).get("rank_weight_map") or {"1":5,"2":3,"3":2,"4":1,"5":1})
    return {str(k): int(v) for k, v in rwm.items()}

def allow_manual_log(uid: str) -> bool:
    """Feature flag to show/hide manual Quick Log. Defaults to False."""
    u = get_user(uid) or {}
    return bool((u.get("prefs") or {}).get("allow_manual_log", False))

def create_goal(user_id: str, title: str, category: str, status: str = "In Progress",
                priority: int = 3, tags: Optional[List[str]] = None) -> str:
    gid = uuid.uuid4().hex[:12]
    now = utc_now_naive()
    doc = {
        "_id": gid, "user": user_id, "title": title.strip(),
        "category": category.strip() or "Other",
        "status": status, "priority": int(priority),
        "is_primary": False,
        "tags": [t.strip() for t in (tags or []) if t.strip()],
        "target_poms": None,
        "created_at": now, "updated_at": now, "schema_version": 1
    }
    db.goals.insert_one(doc)
    st.cache_data.clear()
    return gid

def update_goal(goal_id: str, updates: Dict[str, Any]):
    updates["updated_at"] = utc_now_naive()
    db.goals.update_one({"_id": goal_id, "user": USER_ID}, {"$set": updates})
    st.cache_data.clear()

def delete_goal(goal_id: str) -> bool:
    has_sessions = db.sessions.count_documents({"user": USER_ID, "goal_id": goal_id}) > 0
    if has_sessions:
        return False
    db.weekly_plans.update_many({"user": USER_ID}, {"$pull": {"items": {"goal_id": goal_id}}})
    db.goals.delete_one({"_id": goal_id, "user": USER_ID})
    st.cache_data.clear()
    return True

def upsert_week_plan(uid: str, week_key: str, week_start: str, week_end: str,
                     capacity: Dict[str, int], items: List[Dict[str, Any]]):
    _id = f"{uid}|{week_key}"
    now = utc_now_naive()
    db.weekly_plans.update_one(
        {"_id": _id},
        {"$setOnInsert": {"_id": _id, "user": uid, "created_at": now, "schema_version": 1},
         "$set": {"week_key": week_key, "week_start": week_start, "week_end": week_end,
                  "capacity": capacity, "items": items, "updated_at": now}},
        upsert=True
    )
    st.cache_data.clear()

def upsert_daily_target(uid: str, date_ist: str, target_pomos: int, target_minutes: Optional[int] = None):
    _id = f"{uid}|{date_ist}"
    now = utc_now_naive()
    db.daily_targets.update_one(
        {"_id": _id},
        {"$setOnInsert": {"_id": _id, "user": uid, "date_ist": date_ist, "created_at": now, "schema_version": 1},
         "$set": {"target_pomos": int(target_pomos),
                  "target_minutes": int(target_minutes or target_pomos * 25),
                  "source": "user", "updated_at": now}},
        upsert=True
    )
    st.cache_data.clear()

def get_daily_target(uid: str, date_ist: str) -> Optional[Dict[str, Any]]:
    return db.daily_targets.find_one({"user": uid, "date_ist": date_ist})

def determine_alloc_bucket(uid: str, week_key: str, goal_id: str, planned_current: int) -> str:
    pe_map = aggregate_pe_by_goal_bucket(uid, week_key)
    done_current = pe_map.get(goal_id, {}).get("current", 0.0)
    return "current" if done_current + 1e-6 < float(planned_current) else "backlog"

# ── Hardened insert_session (fixes 'updated_at' conflict)
def insert_session(
    user_id: str,
    t: str,                 # "W" or "B"
    dur_min: int,
    ended_at_ist: datetime,
    *,
    kind: Optional[str] = None,               # "focus" or "activity" (for W)
    activity_type: Optional[str] = None,      # exercise/meditation/breathing/other
    intensity: Optional[str] = None,          # only used in manual flows (we've removed manual logging)
    deep_work: Optional[bool] = None,
    goal_mode: Optional[str] = None,          # "weekly" or "custom"
    goal_id: Optional[str] = None,
    task: Optional[str] = None,
    cat: Optional[str] = None,
    alloc_bucket: Optional[str] = None,
    break_autostart: Optional[bool] = None,
    skipped: Optional[bool] = None,
    post_checkin: Optional[Dict[str, Any]] = None,
    device: Optional[str] = "web"
) -> str:
    t = "B" if str(t).upper() == "B" else "W"
    dur_min = max(1, int(dur_min))
    pe = pom_equiv(dur_min)

    if ended_at_ist.tzinfo is None:
        ended_at_ist = IST.localize(ended_at_ist)
    started_at_ist = ended_at_ist - timedelta(minutes=dur_min)

    started_utc = _to_utc_naive(started_at_ist)
    ended_utc   = _to_utc_naive(ended_at_ist)

    date_ist = started_at_ist.astimezone(IST).date().isoformat()
    week_key = week_key_from_datestr(date_ist)

    kind = (kind or None)
    if kind is not None:
        kind = str(kind).strip().lower()
        if kind not in {"focus", "activity"}:
            kind = "focus" if t == "W" else None

    if kind == "activity":
        # normalize activity fields
        if not activity_type or str(activity_type).strip().lower() not in ALLOWED_ACTIVITY_TYPES:
            activity_type = "other"
        else:
            activity_type = str(activity_type).strip().lower()
        goal_mode = None
        goal_id = None
        alloc_bucket = None
        deep_work = None
        cat = cat or "Wellbeing"
    else:
        activity_type = None

    if kind == "focus" and goal_mode == "weekly" and goal_id:
        if alloc_bucket:
            alloc_bucket = str(alloc_bucket).strip().lower()
            if alloc_bucket not in ALLOWED_BUCKETS:
                alloc_bucket = None
    else:
        goal_mode = ("custom" if (kind == "focus" and t == "W" and not goal_id) else goal_mode)
        alloc_bucket = None

    safe_pc = None
    if isinstance(post_checkin, dict):
        pc = {}
        q = post_checkin.get("quality_1to5")
        m = post_checkin.get("mood_1to5")
        e = post_checkin.get("energy_1to5")
        if isinstance(q, int) and 1 <= q <= 5: pc["quality_1to5"] = q
        if isinstance(m, int) and 1 <= m <= 5: pc["mood_1to5"] = m
        if isinstance(e, int) and 1 <= e <= 5: pc["energy_1to5"] = e
        if post_checkin.get("distraction") is not None:
            pc["distraction"] = str(post_checkin.get("distraction"))
        if post_checkin.get("note") is not None:
            pc["note"] = str(post_checkin.get("note"))
        if pc:
            safe_pc = pc

    sid = f"{user_id}|{date_ist}|{t}|{int(started_at_ist.timestamp())}|{dur_min}"
    now = utc_now_naive()

    # Build full doc then split so updated_at is NOT in $setOnInsert (avoids conflict)
    full_doc = {
        "_id": sid,
        "user": user_id,
        "date_ist": date_ist,
        "week_key": week_key,
        "t": t,
        "dur_min": int(dur_min),
        "pom_equiv": float(pe),
        "started_at_ist": started_utc,
        "ended_at_ist": ended_utc,
        "schema_version": 1,
        "kind": kind,
        "activity_type": activity_type,
        "intensity": (str(intensity).strip().lower() if intensity else None),
        "deep_work": (bool(deep_work) if isinstance(deep_work, bool) else None),
        "context_switch": False,
        "goal_mode": (str(goal_mode).strip().lower() if goal_mode else None),
        "goal_id": goal_id,
        "task": (str(task) if task else None),
        "cat": (str(cat) if cat else None),
        "alloc_bucket": alloc_bucket,
        "break_autostart": (bool(break_autostart) if isinstance(break_autostart, bool) else None),
        "skipped": (bool(skipped) if isinstance(skipped, bool) else None),
        "post_checkin": safe_pc,
        "device": (str(device) if device else None),
        "created_at": now,
        "updated_at": now,
    }
    full_doc = {k: v for k, v in full_doc.items() if v is not None}
    doc_on_insert = dict(full_doc)
    doc_on_insert.pop("updated_at", None)

    try:
        db.sessions.update_one(
            {"_id": sid},
            {"$setOnInsert": doc_on_insert, "$set": {"updated_at": now}},
            upsert=True
        )
    except WriteError as e:
        details = getattr(e, "details", None) or {}
        err = details.get("errmsg") or str(e)
        reasons = details.get("errInfo") or {}
        st.error("❌ Failed to write session (schema validation).")
        with st.expander("Debug details (validator)"):
            st.write({"error": err, "reasons": reasons, "doc_keys": list(full_doc.keys())})
        raise

    # bust caches so UI reflects new data immediately
    st.cache_data.clear()
    return sid

def delete_last_today_session(uid: str, date_ist: str) -> Optional[str]:
    last = db.sessions.find({"user": uid, "date_ist": date_ist}).sort("started_at_ist", -1).limit(1)
    last = next(iter(last), None)
    if last:
        db.sessions.delete_one({"_id": last["_id"]})
        st.cache_data.clear()
        return last["_id"]
    return None

def update_session_post_checkin(sid: str, payload: Dict[str, Any]):
    db.sessions.update_one({"_id": sid, "user": USER_ID},
                           {"$set": {"post_checkin": payload, "updated_at": utc_now_naive()}})
    st.cache_data.clear()

# ── Derived plan from active goals
def derive_auto_plan_from_active(uid: str, week_key: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    wkday, wkend = get_user_capacity_defaults(uid)
    total_capacity = wkday * 5 + wkend * 2
    rwm = get_rank_weight_map(uid)
    goals = [g for g in get_goals(uid) if g.get("status") == "In Progress"]
    if not goals or total_capacity <= 0:
        return ({"weekday": wkday, "weekend": wkend, "total": total_capacity}, [])
    def w_for(g):
        rank = int(g.get("priority", 3))
        return int(rwm.get(str(rank), 1))
    weights = [w_for(g) for g in goals]
    wsum = sum(weights) or 1
    shares = [(w / wsum) * total_capacity for w in weights]
    base = [int(np.floor(s)) for s in shares]
    left = total_capacity - sum(base)
    frac = [s - b for s, b in zip(shares, base)]
    order = np.argsort(-np.array(frac))
    for i in range(int(left)):
        base[order[i]] += 1
    items = []
    for g, pc in zip(goals, base):
        rank = int(g.get("priority", 3))
        items.append({
            "goal_id": g["_id"], "priority_rank": rank,
            "weight": int(rwm.get(str(rank), 1)),
            "planned_current": int(pc),
            "backlog_in": 0,
            "total_target": int(pc),
            "status_at_plan": "In Progress",
            "close_action": None,
            "notes": None
        })
    return ({"weekday": wkday, "weekend": wkend, "total": total_capacity}, items)

# ──────────────────────────────────────────────────────────────────────────────
# Planner helpers (rebalance & move-in)
# ──────────────────────────────────────────────────────────────────────────────
def _planner_rebalance(df: pd.DataFrame, total_capacity: int, rwm: Dict[str, int]) -> pd.DataFrame:
    """Recompute planned_current by weights so the sum equals total_capacity."""
    if df is None or df.empty:
        return df
    m = df.copy()
    if "rank" not in m:
        m["rank"] = "3"
    m["rank"] = m["rank"].astype(str)
    m["weight"] = m["rank"].map(lambda r: int(rwm.get(str(r), 1)))
    wsum = int(m["weight"].sum()) or 1
    shares = (m["weight"] / float(wsum)) * float(total_capacity)
    base = np.floor(shares).astype(int)
    left = int(total_capacity - int(base.sum()))
    frac = (shares - base).values
    order = np.argsort(-frac)
    for i in range(max(0, left)):
        base.iloc[order[i]] += 1
    m["planned_current"] = base.astype(int).values
    m["backlog_in"] = m.get("backlog_in", 0).fillna(0).astype(int)
    m["total_target"] = (m["planned_current"] + m["backlog_in"]).astype(int)
    return m.drop(columns=["weight"], errors="ignore")

def _planner_add_goal_row(buf_key: str, g: Dict[str, Any], rwm: Dict[str,int], total_capacity: int):
    """Append a goal to the planner buffer (if missing) and auto-rebalance."""
    if buf_key not in st.session_state:
        st.session_state[buf_key] = pd.DataFrame(columns=["goal_id","title","category","rank","planned_current","backlog_in","total_target","notes"])
    df = st.session_state[buf_key].copy()

    gid = g["_id"]
    if "goal_id" in df.columns and (df["goal_id"] == gid).any():
        return  # already present

    row = {
        "goal_id": gid,
        "title": g.get("title",""),
        "category": g.get("category",""),
        "rank": str(int(g.get("priority", 3))),
        "planned_current": 0,
        "backlog_in": 0,
        "total_target": 0,
        "notes": ""
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Auto-rebalance to keep capacity respected
    df = _planner_rebalance(df, total_capacity, rwm)
    st.session_state[buf_key] = df

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Connection")
st.sidebar.write(f"**DB:** `{db.name}`")
st.sidebar.write(f"**User:** `{USER_ID}`")

with st.sidebar.expander("🔍 Diagnostics", expanded=False):
    try:
        info = db.command("buildInfo")
        st.write("Connected:", True)
        st.write("Mongo Version:", info.get("version"))
        st.write("Collections:", sorted(db.list_collection_names()))
    except Exception as e:
        st.error(f"Diagnostics failed: {e}")

today = today_iso()
default_week_key = week_key_from_datestr(today)
goals_map = get_goals_map(USER_ID)
default_plan = get_week_plan(USER_ID, default_week_key)
if not default_plan:
    cap_auto, items_auto = derive_auto_plan_from_active(USER_ID, default_week_key)
    default_plan = {
        "_id": f"{USER_ID}|{default_week_key}",
        "user": USER_ID,
        "week_key": default_week_key,
        "week_start": week_dates_list(default_week_key)[0],
        "week_end": week_dates_list(default_week_key)[-1],
        "capacity": cap_auto,
        "items": items_auto,
        "derived": True
    }

st.sidebar.subheader(f"📅 Week {default_week_key}")
if default_plan and default_plan.get("items"):
    st.sidebar.caption(f"{default_plan.get('week_start')} → {default_plan.get('week_end')}")
    cap = default_plan.get("capacity", {})
    st.sidebar.write(f"Capacity: **{cap.get('total', 0)}** poms")
else:
    st.sidebar.info("No weekly plan (and no active goals) for this ISO week yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_timer, tab_planner, tab_analytics = st.tabs(["⏱️ Timer & Log", "🗂️ Weekly Planner", "📈 Analytics"])

# If we flagged a beep previously, play once and clear
if st.session_state.get("beep_once"):
    play_finish_sound()
    st.session_state["beep_once"] = False

def render_activity_heatmap(user_id: str, weeks: int = 52):
    """
    Draw a GitHub-style heatmap showing total WORK minutes per day
    (t == 'W', includes focus + activity) for the last `weeks` weeks.
    """
    # 1) Fetch minutes per day
    agg = db.sessions.aggregate([
        {"$match": {"user": user_id, "t": "W"}},
        {"$group": {"_id": "$date_ist", "mins": {"$sum": "$dur_min"}}}
    ])
    mins_by_date = {pd.to_datetime(doc["_id"]).date(): int(doc["mins"])
                    for doc in agg if doc.get("_id")}

    if not mins_by_date:
        st.info("No activity yet to draw heatmap.")
        return

    # 2) Build date grid (Mon..Sun rows x week columns)
    end = now_ist().date()
    start = end - timedelta(days=weeks * 7 - 1)
    start_monday = start - timedelta(days=start.weekday())   # align to Monday
    cols = ((end - start_monday).days // 7) + 1
    grid = np.zeros((7, cols), dtype=int)

    all_dates = pd.date_range(start=start_monday, end=end, freq="D").to_pydatetime()
    for dt in all_dates:
        d = dt.date()
        c = ((d - start_monday).days) // 7
        r = d.weekday()  # 0=Mon .. 6=Sun
        grid[r, c] = mins_by_date.get(d, 0)

    # 3) Scale color using 95th percentile to avoid outliers dominating
    vals = grid.flatten()
    vmax = 30
    if (vals > 0).any():
        vmax = max(vmax, int(np.percentile(vals[vals > 0], 95)))

    # 4) Plot
    fig_w = max(10, cols * 0.22)  # responsive width
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))
    im = ax.imshow(grid, aspect="auto", cmap="Greens", vmin=0, vmax=vmax, origin="upper")

    # Month labels at the first Monday in each month
    month_positions, month_labels = [], []
    for c in range(cols):
        day0 = start_monday + timedelta(days=c * 7)
        if day0.day <= 7:
            month_positions.append(c)
            month_labels.append(day0.strftime("%b"))
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels, fontsize=8)

    # Show a few weekday labels
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(["Mon", "Wed", "Fri", "Sun"], fontsize=8)

    # Clean frame
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel(f"Last {weeks} weeks", fontsize=9)
    ax.set_ylabel("")

    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cb.set_label("Minutes", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    st.pyplot(fig, use_container_width=True)

# =============================================================================
# TAB 1: Timer & Log
# =============================================================================
with tab_timer:
    st.header("⏱️ Focus Timer")
    st.caption(f"IST Date: **{today}** • ISO Week: **{default_week_key}**")

    # global sound toggle (session-only)
    st.toggle("🔊 Sound", value=st.session_state.get("sound_on", True), key="sound_on", help="Play a sound when the timer completes.")

    # Today target
    st.subheader("🎯 Today’s Target")
    tgt = get_daily_target(USER_ID, today)
    target_val = (tgt or {}).get("target_pomos", None)
    colT1, colT2, colT3 = st.columns([1.1, 0.9, 1])
    with colT1:
        st.metric("Target (poms)", value=target_val if target_val is not None else "—")
        actual_pe = total_day_pe(USER_ID, today)
        st.progress(min(actual_pe / float(target_val or 1), 1.0), text=f"Progress: {actual_pe:.1f} / {target_val or 0} pe")
    with colT2:
        with st.form("target_form", clear_on_submit=False):
            new_target = st.number_input("Set/Update target", min_value=0, max_value=50,
                                         value=int(target_val or 6), step=1, key="target_inp")
            submitted = st.form_submit_button("Save target", use_container_width=True)
            if submitted:
                upsert_daily_target(USER_ID, today, int(new_target))
                st.success("Saved target.")
                st.rerun()
    with colT3:
        todays = list_today_sessions(USER_ID, today)
        focus_cnt = sum(1 for s in todays if s.get("t") == "W" and s.get("kind") != "activity")
        breaks_valid = sum(1 for s in todays if s.get("t") == "B" and (s.get("dur_min",0) >= 4) and not s.get("skipped", False))
        st.metric("Focus sessions", focus_cnt)
        st.metric("Valid breaks", breaks_valid)

    st.divider()

    # ── Live Timer (full width) ───────────────────────────────────────────────
    st.subheader("⏳ Live Timer")

    if "timer" not in st.session_state:
        st.session_state.timer = {
            "running": False, "end_ts": None, "started_at": None, "completed": False,
            "t": "W", "dur_min": 25, "kind": "focus",
            "activity_type": None,
            "deep_work": True, "goal_id": None, "task": None, "cat": None,
            "alloc_bucket": None, "auto_break": True, "break_min": 5
        }
    timer = st.session_state.timer

    # quick start (no form submit required)
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("▶️ Focus 25m", use_container_width=True):
            st.session_state["live_type_choice"] = "Work (focus)"
            st.session_state["live_focus_dur"] = 25
    with qc2:
        if st.button("▶️ Focus 50m", use_container_width=True):
            st.session_state["live_type_choice"] = "Work (focus)"
            st.session_state["live_focus_dur"] = 50
    with qc3:
        if st.button("▶️ Activity 10m", use_container_width=True):
            st.session_state["live_type_choice"] = "Activity"
            st.session_state["live_act_dur"] = 10

    default_idx = 0 if timer.get("kind", "focus") == "focus" else 1
    live_type = st.radio("Type", ["Work (focus)", "Activity"], index=default_idx, horizontal=True, key="live_type_choice")

    # Focus form
    if live_type == "Work (focus)":
        # tiny filter for long goal lists
        filt = st.text_input("Filter goals (title contains…)", key="goal_filter", placeholder="optional")

        with st.form("focus_live_form", clear_on_submit=False):
            dur_live = st.number_input("Work duration (minutes)", min_value=5, max_value=120,
                                       value=int(st.session_state.get("live_focus_dur", timer.get("dur_min", 25))),
                                       step=1, key="live_focus_dur")
            deep_live = (dur_live >= 23)

            labels = []
            choices = []
            plan_src = default_plan or {}
            items_for_pick = plan_src.get("items", [])
            pe_map = aggregate_pe_by_goal_bucket(USER_ID, default_week_key)

            if items_for_pick:
                st.caption("Current week's goals:")
                for it in sorted(items_for_pick, key=lambda x: x.get("priority_rank", 99)):
                    gid = it["goal_id"]; g = goals_map.get(gid, {})
                    title = g.get("title", gid); gcat = g.get("category","—")
                    if filt and filt.lower() not in title.lower():
                        continue
                    planned = int(it.get("planned_current",0))
                    cur_pe = pe_map.get(gid, {}).get("current", 0.0)
                    rem_cur = max(planned - cur_pe, 0)
                    labels.append(f"[P{it.get('priority_rank')}] {title} · {gcat} • current {rem_cur}/{planned} • backlog {it.get('backlog_in',0)}")
                    choices.append((gid, planned))
                if labels:
                    sel_label = st.radio("Pick goal", labels, index=0, key="live_pick_goal")
                    idx = labels.index(sel_label)
                    goal_id, planned_current = choices[idx]
                    alloc_bucket = determine_alloc_bucket(USER_ID, default_week_key, goal_id, planned_current) if planned_current > 0 else None
                    cat = goals_map.get(goal_id, {}).get("category")
                else:
                    goal_id = None; alloc_bucket=None; cat=None
                    st.info("No matching goals. Clear the filter or add goals in Planner.")
            else:
                goal_id = None; alloc_bucket=None; cat=None
                st.info("No active goals found. Create a goal in Weekly Planner.")

            task_text = st.text_input("Optional task note", key="live_task_note")
            auto_break = st.checkbox("Auto-break after Work", value=True, key="live_ab")
            break_min = st.number_input("Break length (min)", 1, 30, value=5, key="live_break_min")

            start_focus = st.form_submit_button("▶️ Start Focus Timer", use_container_width=True)

        if start_focus and not timer["running"]:
            timer.update({
                "running": True, "completed": False, "dur_min": int(dur_live),
                "t": "W",
                "kind": "focus",
                "activity_type": None,
                "deep_work": (dur_live >= 23),
                "goal_id": goal_id,
                "task": task_text, "cat": cat,
                "alloc_bucket": (alloc_bucket if goal_id else None),
                "auto_break": bool(auto_break), "break_min": int(break_min),
                "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=int(dur_live))
            })
            st.rerun()

    # Activity form
    else:
        with st.form("activity_live_form", clear_on_submit=False):
            dur_live = st.number_input("Activity duration (minutes)", min_value=1, max_value=180,
                                       value=int(st.session_state.get("live_act_dur", 10)),
                                       step=1, key="live_act_dur")
            activity_type = st.selectbox("Activity type", ["exercise","meditation","breathing","other"], index=1, key="live_act_type")
            note_text = st.text_input("Optional activity note", key="live_act_note")
            start_activity = st.form_submit_button("▶️ Start Activity Timer", use_container_width=True)

        if start_activity and not timer["running"]:
            timer.update({
                "running": True, "completed": False, "dur_min": int(dur_live),
                "t": "W",
                "kind": "activity",
                "activity_type": activity_type,
                "deep_work": None,
                "goal_id": None,
                "task": note_text, "cat": "Wellbeing",
                "alloc_bucket": None,
                "auto_break": False, "break_min": 0,
                "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=int(dur_live))
            })
            st.rerun()

    # Countdown / running
    if timer["running"]:
        total_secs = max(int(timer["dur_min"]) * 60, 1)
        remaining_secs = max(int((timer["end_ts"] - now_ist()).total_seconds()), 0)
        elapsed_secs = total_secs - remaining_secs
        pct_done = min(max(elapsed_secs / total_secs, 0.0), 1.0)

        rem_m, rem_s = divmod(remaining_secs, 60)
        el_m, el_s = divmod(elapsed_secs, 60)
        started_lbl = timer["started_at"].strftime("%H:%M")
        ends_lbl    = timer["end_ts"].strftime("%H:%M")
        tlabel = "Work (focus)" if (timer["kind"] == "focus") else "Activity"

        st.markdown(
            f"""
            <div style="font-size:1.2rem;margin-bottom:0.25rem;">
              ⏳ <b>{tlabel}</b> — {timer['dur_min']} min
            </div>
            <div style="font-size:2.6rem;font-weight:700;letter-spacing:1px;">
              {rem_m:02d}:{rem_s:02d}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(pct_done, text=f"Elapsed {el_m:02d}:{el_s:02d} • Remaining {rem_m:02d}:{rem_s:02d}")

        meta1, meta2, meta3 = st.columns([1, 1, 1])
        with meta1: st.caption(f"Started: **{started_lbl}**")
        with meta2: st.caption(f"Ends: **{ends_lbl}**")
        with meta3: st.caption(f"Last tick: **{now_ist().strftime('%H:%M:%S')}**")

        colL, colM, colR = st.columns(3)
        stop_now = colL.button("⏹️ Stop / Cancel", use_container_width=True, key="btn_stop_live")
        refresh  = colM.button("🔄 Update now", use_container_width=True, key="btn_refresh_live")
        complete_early = colR.button("✅ Complete now", use_container_width=True, key="btn_complete_live")

        if stop_now:
            timer["running"] = False
            st.warning("Timer canceled.")
            st.rerun()

        if complete_early:
            timer["end_ts"] = now_ist()
            remaining_secs = 0

        if remaining_secs <= 0 and not timer["completed"]:
            ended_at = timer["end_ts"]
            started_at = timer["started_at"]
            dur_min_done = max(1, int(round((ended_at - started_at).total_seconds() / 60.0)))

            sid = insert_session(
                USER_ID, timer["t"], int(dur_min_done), ended_at,
                kind=("focus" if timer["kind"] == "focus" and timer["t"] == "W" else
                      "activity" if timer["kind"] == "activity" and timer["t"] == "W" else None),
                activity_type=(timer["activity_type"] if timer["kind"] == "activity" else None),
                deep_work=(timer["deep_work"] if (timer["kind"] == "focus" and timer["t"] == "W") else None),
                goal_mode=("weekly" if (timer["kind"] == "focus" and timer.get("goal_id") and (default_plan and default_plan.get("items"))) else
                           "custom" if (timer["kind"] == "focus" and timer["t"] == "W") else None),
                goal_id=(timer.get("goal_id") if (timer["kind"] == "focus" and timer["t"] == "W") else None),
                task=(timer.get("task") if (timer["t"] == "W") else None),
                cat=(timer.get("cat") if (timer["kind"] == "focus" and timer["t"] == "W") else
                     "Wellbeing" if timer["kind"] == "activity" else None),
                alloc_bucket=(timer.get("alloc_bucket") if (timer["kind"] == "focus" and timer.get("goal_id")) else None),
                break_autostart=(timer["kind"] == "focus" and timer.get("auto_break", False)) if timer["t"] == "W" else None,
                skipped=False if timer["t"] == "B" else None,
                post_checkin=None, device="web-live"
            )

            st.session_state["beep_once"] = True
            st.session_state["pending_sid"] = sid
            st.session_state["pending_kind"] = timer["kind"]

            timer["completed"] = True
            timer["running"] = False
            st.success(f"Session saved. id={sid}")

            if timer["kind"] != "activity" and timer["auto_break"] and timer["break_min"] > 0:
                timer.update({
                    "running": True, "completed": False,
                    "t": "B", "dur_min": timer["break_min"], "kind": None,
                    "activity_type": None, "deep_work": None,
                    "goal_id": None, "task": None, "cat": None, "alloc_bucket": None,
                    "auto_break": False,
                    "started_at": now_ist(), "end_ts": now_ist() + timedelta(minutes=timer["break_min"])
                })
                st.toast("Auto-break started.", icon="⏱️")
            st.rerun()

        if refresh:
            st.rerun()

        time.sleep(1)
        st.rerun()

    st.divider()

    # Post-check-in prompts (appear only after timer completion)
    if st.session_state.get("pending_sid") and st.session_state.get("pending_kind") in {"focus","activity"}:
        kind_lab = "Work" if st.session_state["pending_kind"] == "focus" else "Activity"
        with st.expander(f"🧠 Post-check-in for your last {kind_lab} session", expanded=True):
            colQ, colM, colE = st.columns(3)
            q = colQ.slider("Quality (1–5)", 1, 5, 4, key="pc_q")
            m = colM.slider("Mood (1–5)", 1, 5, 4, key="pc_m")
            e = colE.slider("Energy (1–5)", 1, 5, 4, key="pc_e")
            note = st.text_input("Quick note (optional)", key="pc_note")
            if st.button("Save check-in", use_container_width=True, key="pc_save"):
                update_session_post_checkin(st.session_state["pending_sid"], {
                    "quality_1to5": int(q), "mood_1to5": int(m),
                    "energy_1to5": int(e), "distraction": None,
                    "note": (note or None)
                })
                st.success("Saved.")
                st.session_state.pop("pending_sid", None)
                st.session_state.pop("pending_kind", None)

    # ---- Quick Log (manual template) ----
    if allow_manual_log(USER_ID):
        with st.expander("🧾 Quick Log (manual entry)", expanded=False):
            # Defaults: end now (IST), today, 25 min
            d_date = st.date_input("End date (IST)", value=now_ist().date(), key="ql_date")
            d_time = st.time_input("End time (IST)", value=now_ist().time().replace(microsecond=0), key="ql_time")
            dur = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=25, step=1, key="ql_dur")

            q_kind = st.radio("Type", ["Work (focus)", "Activity"], horizontal=True, key="ql_kind")
            note = st.text_input("Optional note / task", key="ql_note")

            # Build the list of current-week goals with remaining 'current' poms
            labels, choices = [], []
            plan_src = default_plan or {}
            items_for_pick = plan_src.get("items", [])
            pe_map = aggregate_pe_by_goal_bucket(USER_ID, default_week_key)

            goal_id, planned_current = None, 0
            act_type = None

            if q_kind == "Work (focus)":
                if items_for_pick:
                    st.caption("Current week's goals:")
                    for it in sorted(items_for_pick, key=lambda x: x.get("priority_rank", 99)):
                        gid = it["goal_id"]; g = goals_map.get(gid, {})
                        planned = int(it.get("planned_current", 0))
                        cur_pe = pe_map.get(gid, {}).get("current", 0.0)
                        rem_cur = max(planned - cur_pe, 0)
                        labels.append(f"[P{it.get('priority_rank')}] {g.get('title', gid)} · {g.get('category','—')} • current {rem_cur}/{planned} • backlog {it.get('backlog_in',0)}")
                        choices.append((gid, planned))
                    labels = ["— Unplanned / ad-hoc —"] + labels
                    choices = [("UNPLANNED", 0)] + choices
                    sel = st.selectbox("Attach to goal", labels, index=0, key="ql_goal")
                    idx = labels.index(sel)
                    goal_id, planned_current = choices[idx]
                else:
                    st.info("No goals in this week's plan — will log as Unplanned.")
                    goal_id, planned_current = "UNPLANNED", 0

            else:
                # Activity subtype
                act_type = st.selectbox("Activity type", ["exercise", "meditation", "breathing", "other"], index=1, key="ql_act")

            if st.button("💾 Save session", use_container_width=True, key="ql_save"):
                # Compose ended_at (IST-aware)
                end_naive = datetime.combine(d_date, d_time)
                ended_at_ist = IST.localize(end_naive) if end_naive.tzinfo is None else end_naive.astimezone(IST)

                is_focus = (q_kind == "Work (focus)")

                use_goal = (is_focus and goal_id and goal_id != "UNPLANNED")
                alloc_bucket = determine_alloc_bucket(USER_ID, default_week_key, goal_id, planned_current) if use_goal and planned_current > 0 else None
                cat = (goals_map.get(goal_id, {}).get("category") if use_goal else ("Wellbeing" if not is_focus else None))

                sid = insert_session(
                    USER_ID,
                    "W",                              # we log work/activity rows here (breaks are handled by live timer)
                    int(dur),
                    ended_at_ist,
                    kind=("focus" if is_focus else "activity"),
                    activity_type=(act_type if not is_focus else None),
                    deep_work=(is_focus and int(dur) >= 23),
                    goal_mode=("weekly" if use_goal else "custom"),
                    goal_id=(goal_id if use_goal else None),
                    task=(note or None),
                    cat=cat,
                    alloc_bucket=(alloc_bucket if use_goal else None),
                    break_autostart=None,
                    skipped=None,
                    post_checkin=None,
                    device="manual"
                )
                st.success(f"Saved. id={sid}")
                st.rerun()

    st.subheader("📝 Today’s Sessions")
    todays = list_today_sessions(USER_ID, today)
    if not todays:
        st.info("No sessions logged yet.")
    else:
        def fmt_row(s):
            kindlab = "Work" if s.get("t") == "W" else "Break"
            if s.get("kind") == "activity": kindlab = "Activity"
            when = to_ist_display(s.get("started_at_ist")).strftime("%H:%M")
            goal_title = goals_map.get(s.get("goal_id"), {}).get("title") if s.get("goal_id") else (s.get("task") or "—")
            return {
                "When (IST)": when,
                "Type": kindlab, "Dur (min)": s.get("dur_min"), "PE": s.get("pom_equiv"),
                "Goal/Task": goal_title, "Bucket": s.get("alloc_bucket") or "—",
                "Deep": "✓" if s.get("deep_work") else "—",
            }
        st.dataframe([fmt_row(s) for s in todays], use_container_width=True, hide_index=True)
        if st.button("↩️ Undo last entry", use_container_width=True):
            deleted = delete_last_today_session(USER_ID, today)
            st.warning(f"Deleted last session: {deleted}" if deleted else "Nothing to undo.")
            st.rerun()

# =============================================================================
# TAB 2: Weekly Planner
# =============================================================================
with tab_planner:
    st.header("🗂️ Weekly Planner")

    st.subheader("📅 Build / Edit Weekly Plan")

    # Week range pickers (ISO week key derived from chosen start)
    default_monday = (now_ist() - timedelta(days=now_ist().isoweekday() - 1)).date()
    wk_start_date = st.date_input(
        "Week start (choose your preferred start day)",
        value=default_monday,
        key="wk_start_date"
    )
    wk_end_date = wk_start_date + timedelta(days=6)
    wk = week_key_from_date(wk_start_date)
    st.caption(f"Week range: **{wk_start_date.isoformat()} → {wk_end_date.isoformat()}** • ISO key: **{wk}**")

    # Capacity & weights (from user prefs)
    wkday_default, wkend_default = get_user_capacity_defaults(USER_ID)
    colWCap1, colWCap2 = st.columns(2)
    with colWCap1:
        wkday = st.number_input("Weekday poms (per day)", 0, 20, value=wkday_default)
    with colWCap2:
        wkend = st.number_input("Weekend poms (per day)", 0, 30, value=wkend_default)
    total_capacity = int(wkday)*5 + int(wkend)*2
    st.caption(f"Total capacity: **{total_capacity}** poms.")

    # Auto-rebalance toggle
    auto_rebalance_on_rank_change = st.toggle(
        "⚖️ Auto-rebalance when priority (rank) changes",
        value=True,
        help="When ON, changing a goal's priority will instantly rebalance planned allocations to match capacity."
    )

    # Pull plan + goals to build a base view
    existing = get_week_plan(USER_ID, wk)
    rwm = get_rank_weight_map(USER_ID)

    goals_map_full = get_goals_map(USER_ID)
    active_goals = [g for g in goals_map_full.values() if g.get("status") == "In Progress"]
    existing_items = {it["goal_id"]: it for it in (existing.get("items", []) if existing else [])}

    if not existing_items:
        _, derived_items = derive_auto_plan_from_active(USER_ID, wk)
        base_items = derived_items
    else:
        base_items = []
        for g in active_goals:
            gid = g["_id"]
            ex = existing_items.get(gid)
            if ex:
                base_items.append(ex)
            else:
                rank = int(g.get("priority", 3))
                base_items.append({
                    "goal_id": gid, "priority_rank": rank,
                    "weight": int(rwm.get(str(rank), 1)),
                    "planned_current": 0, "backlog_in": 0, "total_target": 0,
                    "status_at_plan": "In Progress", "close_action": None, "notes": None
                })

    # ── Build the editable planner sheet (stateful) ───────────────────────────
    rows = []
    for it in base_items:
        gid = it["goal_id"]; g = goals_map_full.get(gid, {})
        rows.append({
            "goal_id": gid,
            "title": g.get("title",""),
            "category": g.get("category",""),
            "rank": str(int(it.get("priority_rank", int(g.get("priority",3))))),  # default 3
            "planned_current": int(it.get("planned_current", 0)),
            "backlog_in": int(it.get("backlog_in", 0)),
            "total_target": int(it.get("total_target", int(it.get("planned_current",0))+int(it.get("backlog_in",0)))),
            "notes": it.get("notes") or ""
        })

    buf_key = f"planner_df_{wk}"
    sig_key = f"{buf_key}_sig"
    df_fresh = pd.DataFrame(rows, columns=["goal_id","title","category","rank","planned_current","backlog_in","total_target","notes"])


    # If the set of goals changed (added/removed), reset the buffer to the fresh DF
    goal_sig = tuple(sorted(df_fresh["goal_id"].tolist()))
    if st.session_state.get(sig_key) != goal_sig:
        st.session_state.pop(buf_key, None)
        st.session_state[sig_key] = goal_sig

    # Use buffer if present, else the fresh DF
    df_initial = st.session_state.get(buf_key, df_fresh)

    # Editor
    edited = st.data_editor(
        df_initial,
        column_config={
            "title": st.column_config.TextColumn("Goal"),
            "category": st.column_config.TextColumn("Category"),
            "rank": st.column_config.SelectboxColumn("Priority (1=high)", options=["1","2","3","4","5"], width="small"),
            "planned_current": st.column_config.NumberColumn("Planned (current)", step=1, min_value=0),
            "backlog_in": st.column_config.NumberColumn("Backlog In", step=1, min_value=0),
            "total_target": st.column_config.NumberColumn("Total Target", step=1, min_value=0, disabled=True),
            "notes": st.column_config.TextColumn("Notes"),
        },
        use_container_width=True, hide_index=True, num_rows="fixed"
    )

    # Detect rank change and optionally auto-rebalance
    prev_df = st.session_state.get(buf_key)
    st.session_state[buf_key] = edited.copy()

    if auto_rebalance_on_rank_change and prev_df is not None and not edited.empty and not prev_df.empty:
        try:
            prev_rank_map = dict(zip(prev_df["goal_id"], prev_df["rank"].astype(str)))
            new_rank_map  = dict(zip(edited["goal_id"], edited["rank"].astype(str)))
            rank_changed = any(prev_rank_map.get(gid) != new_rank_map.get(gid) for gid in new_rank_map.keys())
        except Exception:
            rank_changed = False

        if rank_changed:
            m = _planner_rebalance(st.session_state[buf_key], total_capacity, rwm)
            st.session_state[buf_key] = m
            st.toast("Rebalanced allocations to reflect new priorities.", icon="⚖️")
            st.rerun()

    # Controls
    colA1, colA2, colA3, colA4 = st.columns([1,1,1,1])
    auto_go   = colA1.button("⚖️ Auto-allocate")
    clear_plan= colA2.button("🧹 Clear planned_current")
    reset_buf = colA3.button("↩️ Reset from goals")
    save_plan = colA4.button("💾 Save plan")

    # Actions
    if auto_go and not edited.empty:
        m = st.session_state[buf_key].copy()
        m = _planner_rebalance(m, total_capacity, rwm)
        st.session_state[buf_key] = m
        st.success("Auto-allocation applied.")
        st.rerun()

    if clear_plan and not edited.empty:
        m = st.session_state[buf_key].copy()
        m["planned_current"] = 0
        m["backlog_in"] = m["backlog_in"].fillna(0).astype(int)
        m["total_target"] = (m["planned_current"] + m["backlog_in"]).astype(int)
        st.session_state[buf_key] = m
        st.info("Cleared plan allocations.")
        st.rerun()

    if reset_buf:
        st.session_state.pop(buf_key, None)
        st.info("Reset to current goals.")
        st.rerun()

    # Planned sum check uses the buffer (so it matches what you see)
    view_df = st.session_state[buf_key]
    planned_sum = int(view_df["planned_current"].sum()) if not view_df.empty else 0
    st.caption(f"Planned current sum: **{planned_sum}** / capacity **{total_capacity}**")
    if planned_sum != total_capacity:
        st.warning("Sum of planned_current should equal capacity total.")
    else:
        st.success("Planned_current matches capacity total ✅")

    if save_plan and not view_df.empty:
        items = []
        for _, r in view_df.iterrows():
            pc = int(r["planned_current"]); bi = int(r["backlog_in"]); rank = int(r["rank"])
            items.append({
                "goal_id": r["goal_id"],
                "priority_rank": rank,
                "weight": int(rwm.get(str(rank), 1)),
                "planned_current": pc,
                "backlog_in": bi,
                "total_target": pc + bi,
                "status_at_plan": "In Progress",
                "close_action": None,
                "notes": r.get("notes") or None
            })
        cap = {"weekday": int(wkday), "weekend": int(wkend), "total": int(total_capacity)}
        upsert_week_plan(USER_ID, wk, wk_start_date.isoformat(), wk_end_date.isoformat(), cap, items)
        st.success(f"Plan saved for ISO week {wk}.")
        st.rerun()

    st.divider()

    # Current week + rollover
    st.subheader("📊 Current Week Allocation")
    plan_cur = get_week_plan(USER_ID, wk)
    if not plan_cur:
        cap_auto, items_auto = derive_auto_plan_from_active(USER_ID, wk)
        plan_cur = {"items": items_auto, "capacity": cap_auto}
        st.caption("_Showing derived allocation (not saved yet)._")

    if not plan_cur or not plan_cur.get("items"):
        st.info("No allocations yet.")
    else:
        pe_map = aggregate_pe_by_goal_bucket(USER_ID, wk)
        rows = []
        for it in sorted(plan_cur.get("items", []), key=lambda x: x.get("priority_rank", 99)):
            gid = it["goal_id"]
            g = goals_map.get(gid, {})
            planned = int(it["planned_current"])
            cur_pe = pe_map.get(gid, {}).get("current", 0.0)
            back_pe = pe_map.get(gid, {}).get("backlog", 0.0)
            rows.append({
                "Priority": it["priority_rank"],
                "Goal": g.get("title", gid),
                "Category": g.get("category", "—"),
                "Planned": planned,
                "Backlog In": int(it["backlog_in"]),
                "Total Target": int(it["total_target"]),
                "Done Current (pe)": round(cur_pe, 1),
                "Done Backlog (pe)": round(back_pe, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("↩️ Rollover Backlog from Previous Week")
        st.caption("We add (prev total − prev actual) to this week's **Backlog In** per goal, if that goal also exists this week.")
        prev_wk = prev_week_key(wk)
        if st.button(f"Compute & Apply Rollover from {prev_wk}", use_container_width=True):
            prev = get_week_plan(USER_ID, prev_wk)
            if not prev:
                st.warning("No previous week plan found.")
            else:
                carry_map = {}
                # compute per-goal total PE in prev week
                pe_prev = aggregate_pe_by_goal_bucket(USER_ID, prev_wk)
                for it in prev.get("items", []):
                    gid = it["goal_id"]
                    total_target = int(it.get("total_target", 0))
                    actual = int(round(pe_prev.get(gid, {}).get("current", 0.0) + pe_prev.get(gid, {}).get("backlog", 0.0)))
                    carry_map[gid] = max(total_target - actual, 0)

                cur_plan_saved = get_week_plan(USER_ID, wk)
                if not cur_plan_saved:
                    st.warning("Please save the current week plan before applying rollover.")
                else:
                    changed = False
                    new_items = []
                    for it in cur_plan_saved.get("items", []):
                        gid = it["goal_id"]
                        if gid in carry_map:
                            bi = int(carry_map[gid])
                            new_total = int(it.get("planned_current", 0)) + bi
                            if it.get("backlog_in") != bi or it.get("total_target") != new_total:
                                it["backlog_in"] = bi
                                it["total_target"] = new_total
                                changed = True
                        new_items.append(it)
                    if changed:
                        db.weekly_plans.update_one(
                            {"_id": cur_plan_saved["_id"]},
                            {"$set": {"items": new_items, "updated_at": utc_now_naive()}}
                        )
                        st.cache_data.clear()
                        st.success("Rollover applied to current plan.")
                        st.rerun()
                    else:
                        st.info("Rollover computed — no changes needed.")

    st.divider()
    st.subheader("🎯 Goals")

    with st.expander("➕ Add a new goal", expanded=False):
        c1, c2, c3 = st.columns([2,1,1])
        with c1: new_title = st.text_input("Title", key="g_title")
        with c2: new_category = st.selectbox("Category", CATEGORIES, index=0, key="g_cat")
        with c3: new_status = st.selectbox("Status", ["In Progress","On Hold","Completed"], index=0, key="g_status")
        c4, c5 = st.columns([1,2])
        with c4: new_priority = st.selectbox("Priority (1=highest)", options=[1,2,3,4,5], index=2, key="g_priority")
        with c5: new_tags = st.text_input("Tags (comma-separated)", key="g_tags")
        if st.button("Create Goal", type="primary", use_container_width=True):
            if not new_title.strip():
                st.error("Title is required.")
            else:
                gid = create_goal(
                    USER_ID, new_title, new_category, new_status, int(new_priority),
                    [t.strip() for t in (new_tags or "").split(",") if t.strip()]
                )
                st.success(f"Goal created: {gid}")
                st.rerun()

    all_goals = get_goals(USER_ID)
    active_goals    = [g for g in all_goals if (g.get("status") == "In Progress")]
    onhold_goals    = [g for g in all_goals if (g.get("status") == "On Hold")]
    completed_goals = [g for g in all_goals if (g.get("status") == "Completed")]

    colG1, colG2, colG3 = st.columns(3)
    with colG1: st.metric("Active", len(active_goals))
    with colG2: st.metric("On Hold", len(onhold_goals))
    with colG3: st.metric("Completed", len(completed_goals))

    with st.expander("🟢 Active Goals (with weekly progress)", expanded=False):
        current_plan = get_week_plan(USER_ID, wk)
        if not current_plan:
            _, items_auto = derive_auto_plan_from_active(USER_ID, wk)
            planned_by_goal = {it["goal_id"]: int(it["planned_current"]) for it in items_auto}
        else:
            planned_by_goal = {it["goal_id"]: int(it["planned_current"]) for it in current_plan.get("items", [])}

        pe_map = aggregate_pe_by_goal_bucket(USER_ID, wk)

        for g in active_goals:
            gid = g["_id"]
            planned = planned_by_goal.get(gid, 0)
            with st.container(border=True):
                st.write(f"**{g.get('title')}** · _{g.get('category','')}_ · priority {g.get('priority',3)}")
                if planned > 0:
                    cur_pe = pe_map.get(gid, {}).get("current", 0.0)
                    st.progress(min(cur_pe / max(planned,1), 1.0), text=f"Current {cur_pe:.1f} / Planned {planned} pe")
                else:
                    st.caption("Not allocated this week (derived plan may assign 0).")

                cols = st.columns([2,1,1,1,1])
                with cols[0]:
                    etitle = st.text_input("Edit title", value=g.get("title",""), key=f"edit_t_{gid}")
                with cols[1]:
                    estatus = st.selectbox("Status", ["In Progress","On Hold","Completed"],
                                           index={"In Progress":0,"On Hold":1,"Completed":2}.get(g.get("status","In Progress"),0),
                                           key=f"edit_s_{gid}")
                with cols[2]:
                    ecat = st.selectbox("Category", CATEGORIES,
                                        index=max(0, CATEGORIES.index(g.get("category","Learning"))) if g.get("category") in CATEGORIES else 0,
                                        key=f"edit_c_{gid}")
                with cols[3]:
                    eprio = st.selectbox("Priority", options=[1,2,3,4,5], index=int(g.get("priority",3))-1, key=f"edit_p_{gid}")
                with cols[4]:
                    can_delete = db.sessions.count_documents({"user": USER_ID, "goal_id": gid}) == 0
                    del_click = st.button("🗑️ Delete", key=f"del_{gid}", disabled=not can_delete)
                if st.button("Save", key=f"save_{gid}"):
                    update_goal(gid, {"title": etitle, "status": estatus, "category": ecat, "priority": int(eprio)})
                    # If goal is in planner buffer, update its rank and optionally rebalance
                    if auto_rebalance_on_rank_change:
                        _buf_key = f"planner_df_{wk}"
                        if _buf_key in st.session_state:
                            dfp = st.session_state[_buf_key]
                            if not dfp.empty and (dfp["goal_id"] == gid).any():
                                dfp.loc[dfp["goal_id"] == gid, "rank"] = str(int(eprio))
                                st.session_state[_buf_key] = _planner_rebalance(dfp, total_capacity, rwm)
                    st.success("Updated.")
                    st.rerun()
                if del_click:
                    if delete_goal(gid):
                        st.warning("Goal deleted.")
                        st.rerun()
                    else:
                        st.error("This goal has sessions; delete is blocked. Mark it On Hold/Completed instead.")

    with st.expander("⏸️ On Hold", expanded=False):
        for g in onhold_goals:
            with st.container(border=True):
                st.write(f"**{g.get('title')}** · _{g.get('category','')}_ · priority {g.get('priority',3)}")
                cols = st.columns([2,1,1,1,1])
                with cols[0]:
                    etitle = st.text_input("Edit title", value=g.get("title",""), key=f"hold_t_{g['_id']}")
                with cols[1]:
                    ecat = st.selectbox("Category", CATEGORIES,
                                        index=max(0, CATEGORIES.index(g.get("category","Learning"))) if g.get("category") in CATEGORIES else 0,
                                        key=f"hold_c_{g['_id']}")
                with cols[2]:
                    eprio = st.selectbox("Priority", options=[1,2,3,4,5], index=int(g.get("priority",3))-1, key=f"hold_p_{g['_id']}")
                with cols[3]:
                    can_delete = db.sessions.count_documents({"user": USER_ID, "goal_id": g["_id"]}) == 0
                    del_click = st.button("🗑️ Delete", key=f"hold_del_{g['_id']}", disabled=not can_delete)
                with cols[4]:
                    move_click = st.button("↪️ Move to this week's plan", key=f"hold_move_{g['_id']}")

                if st.button("Save", key=f"hold_save_{g['_id']}"):
                    update_goal(g["_id"], {"title": etitle, "category": ecat, "priority": int(eprio)})
                    st.success("Updated.")
                    st.rerun()

                if move_click:
                    # 1) Flip status → In Progress
                    update_goal(g["_id"], {"status": "In Progress"})
                    # 2) Add to planner buffer + auto-rebalance
                    _buf_key = f"planner_df_{wk}"
                    _total_capacity = int(wkday)*5 + int(wkend)*2
                    _planner_add_goal_row(_buf_key, g, rwm, _total_capacity)
                    st.success("Moved to current week and rebalanced.")
                    st.rerun()

                if del_click:
                    if delete_goal(g["_id"]):
                        st.warning("Goal deleted.")
                        st.rerun()
                    else:
                        st.error("This goal has sessions; delete is blocked.")

    with st.expander("✅ Completed", expanded=False):
        for g in completed_goals:
            with st.container(border=True):
                st.write(f"**{g.get('title')}** · _{g.get('category','')}_ · priority {g.get('priority',3)}")
                can_delete = db.sessions.count_documents({"user": USER_ID, "goal_id": g["_id"]}) == 0
                if st.button("🗑️ Delete", key=f"done_del_{g['_id']}", disabled=not can_delete):
                    if delete_goal(g["_id"]):
                        st.warning("Goal deleted.")
                        st.rerun()
                    else:
                        st.error("This goal has sessions; delete is blocked.")

# =============================================================================
# TAB 3: Analytics
# =============================================================================
with tab_analytics:
    st.header("📈 Analytics")

    weeks_sessions = sorted(db.sessions.distinct("week_key", {"user": USER_ID, "t": "W"}))
    weeks_plans    = sorted(db.weekly_plans.distinct("week_key", {"user": USER_ID}))
    all_weeks      = sorted(set(weeks_sessions) | set(weeks_plans))
    count_weeks    = len(all_weeks)

    if count_weeks == 0:
        st.info("No data yet. Log some sessions or save a weekly plan to see analytics.")
    else:
        last_n = choose_last_n("Show last N weeks", available_count=count_weeks, default=6, cap=12, key="lastn_analytics")
        weeks_view = all_weeks[-last_n:] if last_n > 0 else []

        if not weeks_view:
            st.info("No weeks selected.")
        else:
            rows = []
            for W in weeks_view:
                planW = get_week_plan(USER_ID, W)
                planned = sum(int(it.get("planned_current", 0)) for it in (planW.get("items", []) if planW else []))
                pe_doc = next(iter(db.sessions.aggregate([
                    {"$match": {"user": USER_ID, "week_key": W, "t": "W"}},
                    {"$group": {"_id": None, "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
                ])), None)
                actual_pe = float(pe_doc["pe"]) if pe_doc else 0.0
                focus_total = db.sessions.count_documents({"user": USER_ID, "week_key": W, "t": "W", "kind": {"$ne": "activity"}})
                deep = db.sessions.count_documents({"user": USER_ID, "week_key": W, "t": "W", "kind": {"$ne": "activity"}, "deep_work": True})
                deep_pct = pct(deep, focus_total)
                valid_breaks = db.sessions.count_documents({
                    "user": USER_ID, "week_key": W, "t": "B", "dur_min": {"$gte": 4},
                    "$or": [{"skipped": {"$exists": False}}, {"skipped": {"$ne": True}}]
                })
                break_pct = pct(min(valid_breaks, focus_total), focus_total)
                pe_by_mode = {row["_id"]: row["pe"] for row in db.sessions.aggregate([
                    {"$match": {"user": USER_ID, "week_key": W, "t": "W"}},
                    {"$group": {"_id": "$goal_mode", "pe": {"$sum": {"$ifNull": ["$pom_equiv", {"$divide": ["$dur_min", 25.0]}]}}}}
                ])}
                unplan_pct = pct(float(pe_by_mode.get("custom", 0.0)), actual_pe)
                adh = pct(min(actual_pe, planned), planned) if planned else 0.0
                rows.append({
                    "week": W, "planned": planned, "actual_pe": round(actual_pe, 1),
                    "adherence_pct": round(adh, 1), "deep_pct": round(deep_pct, 1),
                    "break_pct": round(break_pct, 1), "unplanned_pct": round(unplan_pct, 1)
                })

            dfw = pd.DataFrame(rows)
            # Surplus column
            dfw["surplus_pe"] = (dfw["actual_pe"] - dfw["planned"]).clip(lower=0)
            st.dataframe(dfw, use_container_width=True, hide_index=True)

            # --- Full-width weekly charts ---
            st.subheader("Adherence % (by week)")
            st.line_chart(dfw.set_index("week")["adherence_pct"])

            st.divider()

            st.subheader("Deep Work % (by week)")
            st.bar_chart(dfw.set_index("week")["deep_pct"])

            st.divider()

            # NEW: GitHub-style daily heatmap (last 52 weeks)
            st.subheader("📆 Daily Activity — GitHub-style heatmap")
            render_activity_heatmap(USER_ID, weeks=52)

            st.divider()

            # Keep a week picker for *other* details below (e.g., Career vs Wellbeing)
            idx_last = max(0, len(weeks_view) - 1)
            sel_week = st.selectbox("Pick a week for details", weeks_view, index=idx_last, key="sel_week_analytics")


            st.divider()
            st.subheader("Career vs Wellbeing (selected week)")
            goal_domain = {}
            for g in db.goals.find({"user": USER_ID}, {"_id":1, "category":1}):
                cat = (g.get("category") or "").lower()
                goal_domain[g["_id"]] = "Wellbeing" if cat in ["health","wellbeing"] else "Career"

            career = 0.0; wellbeing = 0.0
            for s in db.sessions.find({"user": USER_ID, "week_key": sel_week, "t":"W"},
                                      {"goal_id":1,"kind":1,"activity_type":1,"pom_equiv":1,"dur_min":1}):
                pe = s.get("pom_equiv") or (s.get("dur_min",0)/25.0)
                if s.get("kind") == "activity":
                    wellbeing += pe
                else:
                    dom = goal_domain.get(s.get("goal_id"), "Career")
                    if dom == "Wellbeing": wellbeing += pe
                    else: career += pe

            total = career + wellbeing
            if total <= 0:
                st.info("No work recorded in the selected week.")
            else:
                cp = pct(career, total); wp = pct(wellbeing, total)
                fig, ax = plt.subplots()
                ax.pie([career, wellbeing],
                       labels=[f"Career {cp:.1f}%", f"Wellbeing {wp:.1f}%"],
                       autopct="%1.1f%%", startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
st.caption("Focus Timer • Mongo-backed • IST-aware • Planner + Timer + Analytics")
