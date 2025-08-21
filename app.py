import streamlit as st
import time
from datetime import datetime, timedelta, date
import pandas as pd
import pytz
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import hashlib
from typing import List, Dict, Tuple, Optional
import math

# === CONFIG ===
st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Dynamic Weekly Priority & Pomodoro Management"}
)

POMODORO_MIN = 25
BREAK_MIN = 5
IST = pytz.timezone('Asia/Kolkata')
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# === DB INIT ===
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
collection_logs = db["logs"]                 # sessions / targets / notes
users_collection = db["users"]               # user settings
collection_goals = db["goals"]               # weekly goals catalog
collection_plans = db["weekly_plans"]        # plan per week
collection_reflections = db["reflections"]   # daily reflections

def sound_alert():
    """Play a completion sound (browser will auto-play if allowed)."""
    st.components.v1.html(f"""
        <audio autoplay><source src="{SOUND_PATH}" type="audio/mpeg"></audio>
        <script>
            const audio = new Audio('{SOUND_PATH}');
            audio.volume = 0.6;
            audio.play().catch(() => {{ }});
        </script>
    """, height=0)

# === TIME / WEEK HELPERS ===
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> Tuple[date, date]:
    wd = d.weekday()  # 0=Mon
    start = d - timedelta(days=wd)
    end = start + timedelta(days=6)
    return start, end

def week_day_counts(week_start: date) -> Tuple[int, int]:
    wd = we = 0
    for i in range(7):
        day = week_start + timedelta(days=i)
        (wd := wd + 1) if day.weekday() < 5 else (we := we + 1)
    return wd, we

# === UX HELPERS ===
def safe_div(n, d, default=0.0):
    try:
        if d in (None, 0):
            return default
        return float(n) / float(d)
    except Exception:
        return default

def pct_or_dash(n, d, decimals=0):
    if d in (None, 0):
        return "—"
    pct = 100.0 * safe_div(n, d, default=0.0)
    return f"{pct:.{decimals}f}%"

def fmt_minutes(m: Optional[float]) -> str:
    if m is None:
        return "—"
    m = int(round(m))
    if m < 60:
        return f"{m}m"
    h = m // 60
    mm = m % 60
    return f"{h}h {mm}m" if mm else f"{h}h"

def gini_from_counts(counts):
    arr = [c for c in counts if c is not None and c >= 0]
    if not arr:
        return 0.0
    arr = sorted(arr)
    n = len(arr)
    s = sum(arr)
    if s == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(arr, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n

def entropy_norm_from_counts(counts):
    arr = [c for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1:
        return 0.0
    s = float(sum(arr))
    H = -sum((c/s) * math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

def time_to_minutes(tstr: str) -> Optional[int]:
    try:
        dt = datetime.strptime(tstr, "%I:%M %p")
        return dt.hour * 60 + dt.minute
    except Exception:
        return None

def minutes_to_clock(m: int) -> str:
    m = int(m)
    h = (m // 60) % 24
    mm = m % 60
    ampm = "AM" if h < 12 else "PM"
    hh = h if 1 <= h <= 12 else (12 if h % 12 == 0 else h % 12)
    return f"{hh:02d}:{mm:02d} {ampm}"

# === DATA ACCESS / CACHES ===
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
    df["date_only"] = df["date"].dt.date
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

# === GOALS / PLANS ===
def upsert_goal(username: str, title: str, priority_weight: int, goal_type: str, status: str = "New", target_poms: int = 0) -> str:
    gid = hashlib.sha256(f"{username}|{title}".encode()).hexdigest()[:16]
    set_on_insert = {
        "_id": gid,
        "user": username,
        "title": title,
        "target_poms": int(target_poms),
        "poms_completed": int(0),
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
        columns=["_id", "user", "title", "priority_weight", "goal_type", "status", "target_poms", "poms_completed"]
    )

def goal_titles_map(user: str, ids: List[str]) -> Dict[str, str]:
    if not ids:
        return {}
    cur = collection_goals.find({"_id": {"$in": list(ids)}, "user": user}, {"_id": 1, "title": 1})
    return {d["_id"]: d.get("title", "(untitled)") for d in cur}

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
    if d is None:
        d = now_ist().date()
    week_start, week_end = week_bounds_ist(d)
    pid = f"{username}|{week_start.isoformat()}"
    plan = collection_plans.find_one({"_id": pid})
    if plan:
        return plan
    settings = get_user_settings(username)
    wd, we = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(settings, weekdays=wd, weekend_days=we)
    doc = {
        "_id": pid,
        "user": username,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total_poms": total_poms,
        "goals": [],
        "allocations": {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    collection_plans.insert_one(doc)
    return doc

def save_plan_allocations(plan_id: str, goals: List[str], allocations: Dict[str, int]):
    goals_unique = sorted(set(goals))
    clean_alloc = {gid: int(allocations.get(gid, 0)) for gid in goals_unique}
    collection_plans.update_one(
        {"_id": plan_id},
        {"$set": {"goals": goals_unique, "allocations": clean_alloc, "updated_at": datetime.utcnow()}}
    )

# === LOCKING (balance) ===
def is_within_lock_window(plan: Dict, days_window: int = 3) -> bool:
    start = datetime.fromisoformat(plan["week_start"]).date()
    today = now_ist().date()
    return (today - start).days <= (days_window - 1)

@st.cache_data(ttl=30)
def locked_goals_for_user_plan(username: str, plan: Dict, threshold_pct: float = 0.7, min_other: int = 3) -> List[str]:
    if not is_within_lock_window(plan):
        return []
    start = datetime.fromisoformat(plan["week_start"]).date()
    today = now_ist().date()
    df = get_user_sessions(username)
    if df.empty:
        return []
    mask_week = (df["date_only"] >= start) & (df["date_only"] <= today)
    dfw = df[mask_week & (df["pomodoro_type"] == "Work")].copy()
    if dfw.empty:
        return []
    if 'goal_id' not in dfw.columns:
        dfw['goal_id'] = None
    by_goal = dfw.groupby(dfw["goal_id"].astype('object').fillna("NONE")).size().sort_values(ascending=False)
    total = int(by_goal.sum())
    if total < 4:
        return []
    top2 = by_goal.head(2).sum()
    if safe_div(top2, total) >= threshold_pct:
        dominating = list(by_goal.head(2).index)
        dominating = [g for g in dominating if g != "NONE"]
        others = by_goal[~by_goal.index.isin(dominating)]
        if len(others) == 0 or any(others < min_other):
            return dominating
    return []

# === SESSION SAVE ===
def save_pomodoro_session(user: str, is_break: bool, duration: int, goal_id: Optional[str], task: str, category_label: str):
    now = now_ist()
    doc = {
        "type": "Pomodoro",
        "date": now.date().isoformat(),
        "time": now.strftime("%I:%M %p"),
        "pomodoro_type": "Break" if is_break else "Work",
        "duration": duration,
        "user": user,
        "goal_id": goal_id if not is_break else None,
        "task": task if not is_break else "",
        "category": category_label if (category_label and not is_break) else "",
        "created_at": datetime.utcnow()
    }
    collection_logs.insert_one(doc)
    if (not is_break) and goal_id:
        collection_goals.update_one({"_id": goal_id}, {"$inc": {"poms_completed": 1}, "$set": {"updated_at": datetime.utcnow()}})
    get_user_sessions.clear()

# === DAILY TARGETS ===
def get_adaptive_goal(active_days:int):
    if active_days <= 5:
        return 1, "🌱 Building", "Start small – consistency over intensity"
    elif active_days <= 12:
        return 2, "🔥 Growing", "Momentum building – keep going"
    elif active_days <= 19:
        return 3, "💪 Strong", "Push limits – you’re in the zone"
    else:
        return 4, "🚀 Peak", "Excellence mode – maintain the peak"

def save_daily_target(target:int, user:str):
    today = now_ist().date().isoformat()
    target_doc = {"type":"DailyTarget","date": today, "target": int(target), "user": user, "created_at": datetime.utcnow()}
    collection_logs.update_one({"type":"DailyTarget","date": today,"user": user},{"$set": target_doc}, upsert=True)

def get_daily_target(user:str):
    today = now_ist().date().isoformat()
    doc = collection_logs.find_one({"type":"DailyTarget","date": today,"user": user})
    return int(doc["target"]) if doc else None

def render_daily_goal(df: pd.DataFrame):
    if df.empty:
        return 0, 1, 0
    today = now_ist().date()
    df_work = df[df["pomodoro_type"]=="Work"]
    work_today = df_work[df_work["date_only"]==today]
    active_days = len(df_work.groupby(df_work["date_only"]).size())
    today_progress = len(work_today)
    today_minutes = int(work_today['duration'].sum())
    adaptive_goal, _, _ = get_adaptive_goal(active_days)
    return today_progress, adaptive_goal, today_minutes

def render_daily_target_planner(df: pd.DataFrame, today_progress: int):
    st.markdown("## 🎯 Daily Target")
    current_target = get_daily_target(st.session_state.user)
    if df.empty:
        suggested_target, phase_name, _ = 1, "🌱 Building", ""
    else:
        df_work = df[df["pomodoro_type"]=="Work"]
        active_days = len(df_work.groupby(df_work["date_only"]).size())
        suggested_target, phase_name, _ = get_adaptive_goal(active_days)
    col1, col2 = st.columns([2,3])
    with col1:
        if current_target is not None:
            st.info(f"Today’s target: **{current_target}** pomodoros")
            with st.expander("Change today’s target"):
                new_target = st.number_input("New target", 1, 12, value=int(current_target))
                if st.button("Save target"):
                    save_daily_target(int(new_target), st.session_state.user)
                    st.success("Target updated")
                    st.rerun()
        else:
            st.markdown(f"Suggested: **{suggested_target}** ({phase_name})")
            target_input = st.number_input("How many pomodoros today?", 1, 12, value=int(suggested_target))
            if st.button("Set target", use_container_width=True):
                save_daily_target(int(target_input), st.session_state.user)
                st.success("Target set")
                st.rerun()
    with col2:
        if current_target is not None:
            pct = min(100.0, (today_progress / max(1,int(current_target))) * 100)
            st.progress(pct/100.0, text=f"{pct:.0f}% complete")
        else:
            st.info("Set a target to track progress.")

# === ADMIN / UTIL ===
def ensure_indexes():
    try:
        collection_logs.create_index([("user",1),("type",1),("date",1)], name="user_type_date")
        collection_logs.create_index([("goal_id",1)], name="goal_id")
        collection_goals.create_index([("user",1),("title",1)], unique=True, name="user_title_unique")
        collection_plans.create_index([("user",1),("week_start",1)], name="user_week")
        collection_reflections.create_index([("user",1),("date",1)], name="user_date")
        st.success("Indexes checked/created.")
    except Exception as e:
        st.warning(f"Index creation: {e}")

def export_sessions_csv(user: str):
    df = get_user_sessions(user)
    if df.empty:
        st.info("No sessions to export.")
        return
    out = df.sort_values("date")
    st.download_button("⬇️ Export CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"{user}_sessions.csv", mime="text/csv")

# === WEEKLY SNAPSHOT (compact) ===
def render_goal_week_snapshot_section(user: str):
    """At-a-glance weekly status: utilization, run-rate, deep work, per-goal progress."""
    today = now_ist().date()
    week_start, week_end = week_bounds_ist(today)
    plan = get_or_create_weekly_plan(user, week_start)
    allocations = plan.get("allocations", {}) or {}
    total_planned = int(sum(allocations.values())) if allocations else 0

    df = get_user_sessions(user)
    mask_week = (df["date_only"] >= week_start) & (df["date_only"] <= week_end)
    dfw = df[mask_week & (df["pomodoro_type"] == "Work")].copy()

    # Actuals this week
    actual_total = int(len(dfw))
    actual_goals_only = int(dfw["goal_id"].notna().sum())

    # Run-rate to today
    days_passed = (min(today, week_end) - week_start).days + 1
    expected_to_date = int(round(total_planned * safe_div(days_passed, 7)))
    delta = actual_goals_only - expected_to_date

    # Deep work today (≥23m)
    df_today = df[(df["date_only"] == today) & (df["pomodoro_type"] == "Work")]
    deep_today = int((df_today["duration"] >= 23).sum())
    deep_ratio = pct_or_dash(deep_today, len(df_today))

    # Locks & titles
    locked = set(locked_goals_for_user_plan(user, plan))
    goal_ids = list(allocations.keys())
    titles = goal_titles_map(user, goal_ids)

    # Per-goal progress
    actual_by_goal = dfw[dfw["goal_id"].notna()].groupby("goal_id").size().to_dict()
    shortfalls = {gid: max(0, int(allocations.get(gid, 0)) - int(actual_by_goal.get(gid, 0))) for gid in goal_ids}
    next_gid = max(shortfalls, key=lambda g: shortfalls[g]) if shortfalls else None
    if next_gid and shortfalls[next_gid] <= 0:
        next_gid = None

    st.subheader("📌 This Week")
    a, b, c, d = st.columns(4)
    with a:
        st.metric("Capacity Used", pct_or_dash(actual_total, total_planned), help="All sessions vs planned total")
    with b:
        rr_label = "On track" if delta >= 0 else f"{abs(delta)} behind"
        rr_value = f"{max(0, int(actual_goals_only))}/{expected_to_date if total_planned>0 else 0}"
        st.metric("Run-Rate", rr_value, help=f"Goal-linked sessions so far • {rr_label}")
    with c:
        st.metric("Deep Work Today", deep_ratio, help="Share of today’s sessions ≥23 min")
    with d:
        st.metric("Planned", f"{total_planned}", help="Total planned sessions for the week")

    # Suggest next block
    if total_planned == 0:
        st.info("No plan for this week. Use **Weekly Planner** to distribute capacity.")
    elif next_gid:
        st.info(f"Next best block: **{titles.get(next_gid, '(goal)')}** (shortfall {shortfalls[next_gid]}).")
    else:
        st.success("On pace. Continue with any planned goal.")

    # Per-goal mini progress bars
    if allocations:
        st.markdown("#### Goal Progress")
        cols = st.columns(min(4, len(allocations)))
        for i, (gid, planned) in enumerate(allocations.items()):
            with cols[i % len(cols)]:
                title = titles.get(gid, "(goal)")
                actual = int(actual_by_goal.get(gid, 0))
                lock_str = " 🔒" if gid in locked else ""
                st.markdown(f"**{title}{lock_str}**")
                if planned > 0:
                    st.progress(min(1.0, safe_div(actual, planned)),
                                text=f"{actual}/{planned}")
                else:
                    st.caption("No plan set")

# === RHYTHM VISUALS ===
def render_start_time_stability_chart(df_all: pd.DataFrame, days: int = 30):
    """First work start time per day (last N days), with 7-day avg & ±30m band."""
    today = now_ist().date()
    cutoff = today - timedelta(days=days-1)
    dfw = df_all[(df_all["pomodoro_type"]=="Work") & (df_all["date_only"]>=cutoff)].copy()
    if dfw.empty:
        st.info("No recent work sessions for start-time chart.")
        return

    dfw["start_min"] = dfw["time"].apply(time_to_minutes)
    g = dfw.groupby("date_only")["start_min"].min().dropna().sort_index()
    if g.empty:
        st.info("Start times not available.")
        return

    s = g.copy()
    roll = s.rolling(7, min_periods=1).mean()
    band_low = roll - 30
    band_high = roll + 30

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=band_high, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=s.index, y=band_low, mode="lines", fill="tonexty", line=dict(width=0),
                             name="±30m band", hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers",
                             name="First Start", hovertemplate="%{x|%b %d}: %{y} min<extra></extra>"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines",
                             name="7-day Avg", line=dict(dash="dash")))

    tickvals = [6*60 + 60*k for k in range(0, 16)]  # 6AM..9PM
    ticktext = [minutes_to_clock(v) for v in tickvals]
    fig.update_layout(title="Start-time Stability (last 30 days)",
                      height=300, margin=dict(l=10, r=10, t=40, b=30),
                      xaxis_title="", yaxis=dict(title="", tickmode="array",
                                                tickvals=tickvals, ticktext=ticktext))
    st.plotly_chart(fig, use_container_width=True)

def render_consistency_heatmap(df_all: pd.DataFrame, weeks: int = 8):
    """GitHub-style heatmap: rows=Mon..Sun, cols=weeks (last N)."""
    if df_all.empty:
        return
    today = now_ist().date()
    this_monday = today - timedelta(days=today.weekday())
    mondays = [this_monday - timedelta(weeks=w) for w in range(weeks-1, -1, -1)]
    z = []
    for dow in range(7):  # Mon..Sun
        row = []
        for m in mondays:
            d = m + timedelta(days=dow)
            cnt = int(((df_all["date_only"] == d) & (df_all["pomodoro_type"]=="Work")).sum())
            row.append(cnt)
        z.append(row)
    y_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    x_labels = [m.strftime("%b %d") for m in mondays]
    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale="YlGn",
                                    colorbar=dict(title="Sessions")))
    fig.update_layout(title=f"Consistency (last {weeks} weeks)",
                      height=240, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# === WEEKLY PLANNER ===
def render_weekly_planner():
    st.header("📅 Weekly Planner")
    user = st.session_state.user
    pick_date = st.date_input("Plan week of", value=st.session_state.planning_week_date)
    week_start, week_end = week_bounds_ist(pick_date)
    if pick_date != st.session_state.planning_week_date:
        st.session_state.planning_week_date = pick_date
        st.rerun()

    settings = get_user_settings(user)
    plan = get_or_create_weekly_plan(user, week_start)

    colA, colB, colC = st.columns(3)
    with colA:
        wp = st.number_input("Weekday capacity", 0, 12, value=settings["weekday_poms"],
                             help="Planned pomodoros per weekday")
    with colB:
        we = st.number_input("Weekend capacity", 0, 12, value=settings["weekend_poms"],
                             help="Planned pomodoros per weekend day")
    with colC:
        wd_count, we_count = week_day_counts(week_start)
        total = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we},
                                        weekdays=wd_count, weekend_days=we_count)
        st.metric(f"Capacity {week_start} → {week_end}", f"{total}")
        if (wp != settings["weekday_poms"]) or (we != settings["weekend_poms"]):
            if st.button("Save defaults", use_container_width=True):
                users_collection.update_one({"username": user},
                                            {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
                get_user_settings.clear()
                st.success("Defaults updated")
                st.rerun()

    st.divider()
    st.subheader("🎯 Goals")
    with st.expander("Add / Update Goal"):
        g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
        g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        g_weight = st.select_slider("Priority", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
        g_status = st.selectbox("Status", ["New","In Progress","Completed","On Hold","Archived"], index=0)
        if st.button("Save goal"):
            if g_title.strip():
                upsert_goal(user, g_title.strip(), int(g_weight), g_type, g_status)
                fetch_goals.clear()
                st.success("Goal saved")
                st.rerun()
            else:
                st.warning("Please provide a title")

    goals_df = fetch_goals(user, statuses=["New","In Progress"])
    if goals_df.empty:
        st.info("Add 3–4 goals to plan your week.")
        return

    st.dataframe(
        goals_df[["_id","title","goal_type","priority_weight","status"]]
        .rename(columns={"_id":"Goal ID","goal_type":"Type","priority_weight":"Priority"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("🧮 Allocate Weekly Pomodoros")
    wd_count, we_count = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(settings, weekdays=wd_count, weekend_days=we_count)
    weight_map = {row["_id"]: int(row["priority_weight"]) for _, row in goals_df.iterrows()}
    auto = proportional_allocation(total_poms, weight_map)
    st.caption("Adjust the numbers below – the total should match capacity.")

    prev_start = week_start - timedelta(days=7)
    prev_plan = collection_plans.find_one({"_id": f"{user}|{prev_start.isoformat()}"})

    plan_has_alloc = bool(plan.get("allocations"))
    if plan_has_alloc:
        st.warning("A plan already exists for this week. Tick **Overwrite** to replace it entirely.")

    if (not plan.get("allocations")) and prev_plan and prev_plan.get("allocations"):
        if st.button(f"Copy last week ({prev_start} → {prev_start+timedelta(days=6)})"):
            save_plan_allocations(plan["_id"], list(prev_plan["allocations"].keys()), prev_plan["allocations"])
            st.success("Copied last week")
            st.rerun()

    edited = {}
    cols = st.columns(min(4, len(auto)) if len(auto) > 0 else 1)
    for i, (_, row) in enumerate(goals_df.iterrows()):
        with cols[i % len(cols)]:
            default_val = int(plan.get("allocations", {}).get(row['_id'], auto[row['_id']]))
            val = st.number_input(f"{row['title']}", min_value=0, max_value=total_poms,
                                  value=default_val, step=1, key=f"alloc_{row['_id']}")
            edited[row["_id"]] = int(val)

    sum_edit = sum(edited.values())
    if sum_edit != total_poms:
        st.warning(f"Total is {sum_edit} (should be {total_poms}). Click to auto-balance.")
        if st.button("Auto-balance"):
            edited = proportional_allocation(total_poms, {gid: max(1, v) for gid, v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.rerun()

    overwrite = st.checkbox("Overwrite existing plan", value=False, disabled=not plan_has_alloc)
    btn_label = "Update Plan" if plan_has_alloc else "Save Plan"
    disabled_save = plan_has_alloc and (not overwrite)

    if st.button(btn_label, type="primary", disabled=disabled_save):
        save_plan_allocations(plan["_id"], list(edited.keys()), edited)
        st.success("Plan saved")
        st.rerun()

# === TIMER WIDGET ===
def render_timer_widget(auto_break: bool) -> bool:
    if not st.session_state.start_time:
        return False
    duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
    remaining = int(st.session_state.start_time + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break" if st.session_state.is_break else f"Working: {st.session_state.task or '—'}"
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
        progress = 1 - (remaining/duration)
        st.progress(progress)
        st.info("Relax." if st.session_state.is_break else "Stay focused.")
        time.sleep(1)
        st.rerun()
        return True
    else:
        # complete
        label = st.session_state.active_goal_title
        was_break = st.session_state.is_break
        save_pomodoro_session(
            user=st.session_state.user,
            is_break=was_break,
            duration=BREAK_MIN if was_break else POMODORO_MIN,
            goal_id=st.session_state.active_goal_id,
            task=st.session_state.task,
            category_label=label
        )
        sound_alert()
        st.balloons(); st.success("Session complete")

        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""

        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

# === FOCUS TIMER ===
def render_focus_timer():
    st.header("🎯 Focus Timer")

    settings = get_user_settings(st.session_state.user)
    colset1, _ = st.columns([1, 3])
    with colset1:
        auto_break_ui = st.toggle("Auto-start break", value=settings.get("auto_break", True),
                                  help="Automatically start a 5-minute break after each work session")
        if auto_break_ui != settings.get("auto_break", True):
            users_collection.update_one({"username": st.session_state.user}, {"$set": {"auto_break": bool(auto_break_ui)}})
            get_user_settings.clear()

    if render_timer_widget(auto_break=get_user_settings(st.session_state.user).get("auto_break", True)):
        return

    user = st.session_state.user
    plan = get_or_create_weekly_plan(user, now_ist().date())

    # Daily target
    df_all = get_user_sessions(user)
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df_all)
    render_daily_target_planner(df_all, today_progress)
    st.divider()

    # Weekly snapshot + rhythm
    render_goal_week_snapshot_section(user)
    with st.expander("📈 Rhythm", expanded=False):
        c1, c2 = st.columns([3,2])
        with c1:
            render_start_time_stability_chart(df_all, days=30)
        with c2:
            render_consistency_heatmap(df_all, weeks=8)
    st.divider()

    # Mode
    mode = st.radio("Session Type", ["Weekly Goal", "Custom/Unplanned"], horizontal=True)

    if mode == "Weekly Goal":
        active_goal_ids = plan.get("goals", [])
        goals_df = fetch_goals(user, statuses=["New","In Progress"])
        goals_df = goals_df[goals_df["_id"].isin(active_goal_ids)] if not goals_df.empty else goals_df

        locked = set(locked_goals_for_user_plan(user, plan))
        if locked:
            st.warning("Balance in progress: top goals are temporarily locked. Log a few on other goals to unlock.")

        choices = []
        for _, r in goals_df.iterrows():
            title, gid = r['title'], r['_id']
            disabled = gid in locked
            alloc = plan.get('allocations', {}).get(gid, 0)
            label = f"{title}  ·  plan:{alloc}"
            choices.append((label, gid, disabled))

        c1, c2 = st.columns([1,2])
        with c1:
            options_labels = [lab + ("  🔒" if dis else "") for (lab,_,dis) in choices] or ["(no goals)"]
            selected_idx = st.selectbox("Goal", options=range(len(options_labels)),
                                        format_func=lambda i: options_labels[i], disabled=len(choices)==0)
            selected_gid = choices[selected_idx][1] if choices else None
            selected_title = choices[selected_idx][0].split('  ·')[0] if choices else ""
        with c2:
            task = st.text_input("Task (micro)", placeholder="e.g., Revise Unit-2 notes")

        st.session_state.active_goal_id = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = (not task.strip()) or (selected_gid in locked if selected_gid else False) or (len(choices)==0)
            if st.button("▶️ Start 25m", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
            if disabled and len(choices)>0 and selected_gid in locked:
                st.caption("Locked for balance. Pick another goal for now.")
        with colb:
            if st.button("☕ Break 5m", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()
    else:
        # Custom / Unplanned
        current_cats = settings.get("custom_categories", ["Learning","Projects","Research","Planning"])
        cat_options = current_cats + ["+ Add New"]
        selected = st.selectbox("Category", cat_options)
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("Add"):
                if new_cat not in current_cats:
                    users_collection.update_one({"username": user}, {"$addToSet": {"custom_categories": new_cat}})
                    get_user_settings.clear()
                    st.success("Category added")
                    st.rerun()
            category_label = new_cat if new_cat else ""
        else:
            category_label = selected
        task = st.text_input("Task (micro)", placeholder="e.g., Draft outreach emails")
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = category_label
        st.session_state.task = task
        colw, colb = st.columns(2)
        with colw:
            disabled = not (category_label and task.strip())
            if st.button("▶️ Start 25m", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break 5m", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()

    # Today summary
    df = get_user_sessions(user)
    if not df.empty:
        today = now_ist().date()
        today_data = df[df["date_only"] == today]
        work_today = today_data[today_data["pomodoro_type"]=="Work"]
        breaks_today = len(today_data[today_data["pomodoro_type"]=="Break"])
        st.divider(); st.subheader("📊 Today")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("Work Sessions", len(work_today))
        with col2:
            st.metric("Focus Time", fmt_minutes(work_today['duration'].sum()))
        with col3:
            if len(work_today) == 0 and breaks_today == 0:
                ratio_str = "—"
            elif breaks_today == 0:
                ratio_str = "No breaks"
            else:
                ratio_str = f"{safe_div(len(work_today), breaks_today):.2f}"
            st.metric("Work/Break", ratio_str, help="Higher = more work blocks per break")
        with col4:
            current_target = get_daily_target(user)
            if current_target:
                pct = (len(work_today)/max(1,int(current_target)))*100
                st.metric("Target Progress", f"{pct:.0f}%")
            else:
                st.metric("Target Progress", "—")

# === JOURNAL (Notes + Reflection) ===
def render_journal():
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Add Note", "Browse Notes"])
    user = st.session_state.user
    today_iso = now_ist().date().isoformat()

    with tab1:
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form", clear_on_submit=True):
            aligned = st.selectbox("Aligned with plan?", ["Yes","Partly","No"])
            rating = st.slider("Focus quality", 1, 5, 3)
            blockers = st.text_area("Blockers / distractions")
            notes = st.text_area("Insights / notes")
            if st.form_submit_button("Save"):
                collection_reflections.update_one(
                    {"user": user, "date": today_iso},
                    {"$set": {"user": user, "date": today_iso, "aligned": aligned,
                              "focus_rating": int(rating), "blockers": blockers.strip(),
                              "notes": notes.strip(), "created_at": datetime.utcnow()}},
                    upsert=True
                )
                st.success("Saved")
        recs = list(collection_reflections.find({"user": user}).sort("date", -1).limit(14))
        if recs:
            st.subheader("Recent")
            df = pd.DataFrame(recs)
            st.dataframe(df[["date","aligned","focus_rating","blockers","notes"]],
                         use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Add Note")
        with st.form("note_form", clear_on_submit=True):
            c1, c2 = st.columns([1,3])
            with c1:
                d = st.date_input("Date", now_ist())
            with c2:
                content = st.text_area("Note", height=140)
            if st.form_submit_button("Save"):
                if content.strip():
                    nid = hashlib.sha256(f"{d.date().isoformat()}_{content}_{user}".encode()).hexdigest()
                    doc = {"_id": nid, "type":"Note", "date": d.date().isoformat(),
                           "content": content.strip(), "user": user, "created_at": datetime.utcnow()}
                    collection_logs.update_one({"_id": nid}, {"$set": doc}, upsert=True)
                    st.success("Saved")
                else:
                    st.warning("Please write something.")

    with tab3:
        st.subheader("Browse")
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("From", now_ist().date()-timedelta(days=7))
        with c2:
            end = st.date_input("To", now_ist().date())
        q = {"type":"Note","user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
        notes = list(collection_logs.find(q).sort("date", -1))
        if notes:
            for n in notes:
                st.subheader(f"📅 {n['date']}")
                st.write(n['content'])
                st.divider()
        else:
            st.info("No notes in this range.")

# === ANALYTICS & REVIEW ===
def render_analytics_review():
    st.header("📊 Analytics & Review")
    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0)

    user = st.session_state.user
    df_all = get_user_sessions(user)
    if df_all.empty:
        st.info("No sessions yet. Start a pomodoro to populate analytics.")
        return

    df_all["date_only"] = df_all["date"].dt.date
    df_work = df_all[df_all["pomodoro_type"] == "Work"].copy()
    df_break = df_all[df_all["pomodoro_type"] == "Break"].copy()

    if mode == "Week Review":
        pick_date = st.date_input("Review week of", value=st.session_state.review_week_date)
        if pick_date != st.session_state.review_week_date:
            st.session_state.review_week_date = pick_date
            st.rerun()
        week_start, week_end = week_bounds_ist(pick_date)
        plan = get_or_create_weekly_plan(user, week_start)
        planned_alloc = plan.get("allocations", {}) or {}
        total_planned = int(sum(planned_alloc.values())) if planned_alloc else 0

        mask_week = (df_all["date_only"] >= week_start) & (df_all["date_only"] <= week_end)
        dfw = df_work[mask_week].copy()
        dfb = df_break[mask_week].copy()

        work_goal = dfw[dfw["goal_id"].notna()].copy()
        work_custom = dfw[dfw["goal_id"].isna()].copy()
        deep = len(dfw[dfw["duration"] >= 23])
        goal_counts = work_goal.groupby("goal_id").size().values.tolist()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Plan Adherence", pct_or_dash(len(work_goal), total_planned),
                      help="Goal-linked sessions vs planned")
        with c2:
            st.metric("Capacity Used", pct_or_dash(len(dfw), total_planned),
                      help="All sessions vs planned total")
        with c3:
            st.metric("Deep Work", pct_or_dash(deep, len(dfw)),
                      help="Share of sessions ≥23 min")
        with c4:
            st.metric("Balance (Entropy)", f"{entropy_norm_from_counts(goal_counts):.2f}",
                      help="0=one goal dominates, 1=even spread")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Gini (Goals)", f"{gini_from_counts(goal_counts):.2f}",
                      help="0=equal, 1=unequal distribution")
        with c6:
            st.metric("Custom Share", pct_or_dash(len(work_custom), len(dfw)),
                      help="Custom/Unplanned sessions")
        with c7:
            expected_breaks = len(dfw)
            skip = max(0, expected_breaks - len(dfb))
            st.metric("Break Skip", pct_or_dash(skip, expected_breaks),
                      help="Expected 1 break per work session")
        with c8:
            extend = max(0, len(dfb) - expected_breaks)
            st.metric("Break Extend", pct_or_dash(extend, expected_breaks),
                      help="Extra breaks beyond 1:1")

        st.caption(
            f"Avg break: {fmt_minutes(dfb['duration'].mean() if len(dfb) else 0)} • "
            f"Weekly capacity: {total_planned or '—'} "
            f"(weekday {get_user_settings(user)['weekday_poms']}, weekend {get_user_settings(user)['weekend_poms']})"
        )

        st.subheader("Discipline & Rhythm")
        dfw_sorted = dfw.sort_values(["date", "time"])
        switches = 0
        runs = 0
        prev_key = None
        for _, r in dfw_sorted.iterrows():
            key = r["goal_id"] if pd.notna(r["goal_id"]) else f"CAT::{r.get('category','')}"
            if prev_key is not None and key != prev_key:
                switches += 1
            prev_key = key
            runs += 1
        switch_idx = safe_div(switches, max(1, runs - 1))

        starts = [time_to_minutes(x) for x in dfw["time"].tolist() if isinstance(x, str)]
        starts = [s for s in starts if s is not None]
        start_sigma = (pd.Series(starts).std() if len(starts) >= 2 else None)
        wb_ratio = "—"
        if len(dfw) == 0 and len(dfb) == 0:
            wb_ratio = "—"
        elif len(dfb) == 0:
            wb_ratio = "No breaks"
        else:
            wb_ratio = f"{safe_div(len(dfw), len(dfb)):.2f}"
        if len(starts) > 0:
            hours = [s//60 for s in starts]
            peak_hour = pd.Series(hours).mode().iloc[0]
            ampm = "AM" if peak_hour < 12 else "PM"
            ph_disp = f"{(peak_hour if 1 <= peak_hour <= 12 else (12 if peak_hour%12==0 else peak_hour%12))}{ampm}"
        else:
            ph_disp = "—"
        med_sessions_per_task = dfw.groupby("task").size().median() if not dfw.empty else None

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Switching-Cost", f"{switch_idx*100:.0f}%",
                      help="Percent of transitions that switch goal/category")
        with d2:
            st.metric("Start-time σ", f"{start_sigma:.0f} min" if start_sigma is not None else "—",
                      help="Std-dev of start times (smaller is steadier)")
        with d3:
            st.metric("Work/Break", wb_ratio, help="Work blocks per break")
        with d4:
            st.metric("Peak Hour", ph_disp, help="Most common hour you start")

        if not dfw.empty:
            off_blocks = len(dfw[(dfw["duration"] < 20) | (dfw["duration"] > 30)])
            st.caption("Durations look consistent" if off_blocks == 0 else f"Note: {off_blocks} sessions outside 25±5 min")
        st.caption(f"Task granularity (median sessions per task): {med_sessions_per_task:.1f}" if med_sessions_per_task is not None else "Task granularity: —")

        st.divider()
        st.subheader("Per-Goal: Plan vs Actual")

        def title_of(gid):
            if gid is None:
                return "Custom/Unplanned"
            doc = collection_goals.find_one({"_id": gid})
            return doc["title"] if doc else "(missing)"

        planned_df = pd.DataFrame([{"goal_id": gid, "planned": int(v), "title": title_of(gid)} for gid, v in planned_alloc.items()])
        actual_df = dfw.groupby(dfw["goal_id"]).size().rename("actual").reset_index()
        actual_df["title"] = actual_df["goal_id"].apply(title_of)
        merged = pd.merge(planned_df, actual_df, on=["goal_id","title"], how="outer").fillna(0)
        merged["planned"] = merged["planned"].astype(int)
        merged["actual"] = merged["actual"].astype(int)
        if merged["goal_id"].isna().any():
            merged.loc[merged["goal_id"].isna(), "goal_id"] = None

        cA, cB = st.columns([3,2])
        with cA:
            if not merged.empty:
                fig = px.bar(merged.sort_values("planned", ascending=False),
                             x="title", y=["planned","actual"], barmode="group",
                             title="Planned vs Actual")
                fig.update_layout(height=360, xaxis_title="", legend_title="")
                st.plotly_chart(fig, use_container_width=True)
        with cB:
            total_actual_goals = int(work_goal.shape[0])
            carry = max(0, total_planned - total_actual_goals)
            st.metric("Planned", total_planned or 0)
            st.metric("Actual (goals)", total_actual_goals)
            st.metric("Carryover", carry)
            st.metric("Carryover Rate", pct_or_dash(carry, total_planned))

        # Diagnostics (optional)
        if st.toggle("Show diagnostics", value=False):
            missing_link_pct = pct_or_dash(len(dfw[dfw["goal_id"].isna() & (len(planned_alloc) > 0)]), len(dfw))
            st.caption(f"Missing Link (no goal_id within planned week): {missing_link_pct}")

        st.divider()
        if total_planned > 0:
            days = pd.date_range(start=pd.to_datetime(week_start), end=pd.to_datetime(min(week_end, now_ist().date())))
            dfw_goal = dfw[dfw["goal_id"].notna()].copy()
            actual_cum, exp_cum = [], []
            for i, ts in enumerate(days):
                cutoff = ts.date()
                actual_to_d = int((dfw_goal["date_only"] <= cutoff).sum())
                expected_to_d = int(round(total_planned * ((i + 1) / len(days))))
                actual_cum.append(actual_to_d)
                exp_cum.append(expected_to_d)
            rr_df = pd.DataFrame({"day": [ts.strftime("%a %d") for ts in days],
                                  "Expected": exp_cum, "Actual": actual_cum})
            fig_rr = px.line(rr_df, x="day", y=["Expected","Actual"], markers=True, title="Run-Rate (goals only)")
            fig_rr.update_layout(height=330, legend_title="")
            st.plotly_chart(fig_rr, use_container_width=True)

        has_planned = bool(planned_alloc)
        has_goal_actuals = (work_goal.shape[0] > 0)
        if has_planned or has_goal_actuals:
            st.divider()
            st.subheader("Close Out & Rollover")
            if not merged.empty:
                for _, row in merged.sort_values("title").iterrows():
                    gid = row["goal_id"]
                    if gid in (None, "NONE"):
                        continue
                    col1, col2, col3, col4 = st.columns([3,2,2,2])
                    with col1:
                        st.write(f"**{row['title']}**")
                    with col2:
                        status = st.selectbox("Status",
                                              ["Completed","Rollover","On Hold","Archived","In Progress"],
                                              index=4, key=f"close_{gid}_{week_start}")
                    with col3:
                        carry = max(0, int(row["planned"]) - int(row["actual"]))
                        carry = st.number_input("Carry forward", 0, 200, value=carry, key=f"carry_{gid}_{week_start}")
                    with col4:
                        if st.button("Apply", key=f"apply_{gid}_{week_start}"):
                            new_status = ("Completed" if status=="Completed" else
                                          "On Hold" if status=="On Hold" else
                                          "Archived" if status=="Archived" else
                                          "In Progress")
                            collection_goals.update_one({"_id": gid}, {"$set": {"status": new_status}})
                            if status=="Rollover" and carry>0:
                                next_start = week_start + timedelta(days=7)
                                next_plan = get_or_create_weekly_plan(user, next_start)
                                next_alloc = next_plan.get("allocations", {}) or {}
                                next_goals = set(next_plan.get("goals", []))
                                next_goals.add(gid)
                                next_alloc[gid] = next_alloc.get(gid, 0) + int(carry)
                                collection_plans.update_one({"_id": next_plan["_id"]},
                                    {"$set": {"goals": list(next_goals), "allocations": next_alloc,
                                              "updated_at": datetime.utcnow()}})
                            st.success("Updated")
        else:
            st.info("Nothing to close yet. Log some goal-linked sessions first.")

    else:
        # TRENDS
        today = now_ist().date()
        st.subheader("Totals")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", len(df_work))
        with col2:
            st.metric("Total Time", fmt_minutes(df_work["duration"].sum()))
        with col3:
            st.metric("Active Days", int(df_work.groupby("date_only").size().shape[0]))
        with col4:
            avg_daily = df_work.groupby("date_only").size().mean() if len(df_work) else 0
            st.metric("Avg Daily", f"{avg_daily:.1f}")

        st.divider()
        st.subheader("Daily Minutes (30 days)")
        daily_data = []
        for i in range(30):
            d = today - timedelta(days=29 - i)
            dwork = df_work[df_work["date_only"] == d]
            daily_data.append({"date": d.strftime("%m/%d"), "minutes": int(dwork["duration"].sum())})
        daily_df = pd.DataFrame(daily_data)
        if daily_df["minutes"].sum() > 0:
            fig = px.bar(daily_df, x="date", y="minutes", title="Daily Focus Minutes",
                         color="minutes", color_continuous_scale="Blues")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Categories")

        time_filter = st.selectbox("Time window", ["Last 7 days", "Last 30 days", "All time"], index=1)
        if time_filter == "Last 7 days":
            cutoff = today - timedelta(days=7)
            fw = df_work[df_work["date_only"] >= cutoff]
        elif time_filter == "Last 30 days":
            cutoff = today - timedelta(days=30)
            fw = df_work[df_work["date_only"] >= cutoff]
        else:
            fw = df_work

        if fw.empty:
            st.info("No data in this window.")
            return

        cat_stats = fw.groupby("category").agg(duration=("duration", "sum"),
                                               sessions=("duration", "count")).sort_values("duration", ascending=False)
        colA, colB = st.columns([3,2])
        with colA:
            total_time = cat_stats["duration"].sum()
            fig_donut = px.pie(values=cat_stats["duration"], names=cat_stats.index,
                               title=f"Time by Category ({time_filter})",
                               hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            center_text = fmt_minutes(total_time)
            fig_donut.add_annotation(text=f"<b>Total</b><br>{center_text}", x=0.5, y=0.5, showarrow=False)
            fig_donut.update_layout(height=400, showlegend=True, title_x=0.5)
            st.plotly_chart(fig_donut, use_container_width=True)
        with colB:
            st.markdown("#### Performance")
            view = cat_stats.copy()
            view["Time"] = view["duration"].apply(fmt_minutes)
            view["Avg/Session"] = (view["duration"] / view["sessions"]).round(1).astype(str) + "m"
            st.dataframe(view[["Time","sessions","Avg/Session"]],
                         use_container_width=True, hide_index=False,
                         height=min(len(view)*35+38, 300))

        st.subheader("Tasks")
        tstats = fw.groupby(["category","task"]).agg(total_minutes=("duration","sum"),
                                                     sessions=("duration","count")).reset_index()
        tstats = tstats.sort_values("total_minutes", ascending=False)
        colC, colD = st.columns([3,2])
        with colC:
            top_tasks = tstats.head(12)
            if not top_tasks.empty:
                fig_tasks = px.bar(top_tasks, x="total_minutes", y="task", color="category",
                                   title=f"Top Tasks ({time_filter})",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                fig_tasks.update_layout(height=max(400, len(top_tasks)*30),
                                        yaxis={"categoryorder":"total ascending"},
                                        title_x=0.5, showlegend=True)
                st.plotly_chart(fig_tasks, use_container_width=True)
        with colD:
            st.markdown("#### Insights")
            if not tstats.empty:
                total_time = tstats["total_minutes"].sum()
                top = tstats.iloc[0]
                share = safe_div(top["total_minutes"], total_time) * 100
                if share > 50:
                    st.warning("One task dominates your time. Consider rebalancing.")
                elif share > 25:
                    st.info("Clear primary task this period.")
                else:
                    st.success("Time is well distributed across tasks.")

        st.divider()
        st.subheader("Consistency")
        counts_by_day = df_work.groupby("date_only").size().to_dict()
        active_days = len(counts_by_day)
        min_sessions = 1 if active_days <= 12 else 2

        cur_streak = 0
        for i in range(365):
            d = today - timedelta(days=i)
            if counts_by_day.get(d, 0) >= min_sessions:
                cur_streak += 1
            else:
                break
        best, temp = 0, 0
        for i in range(365):
            d = today - timedelta(days=i)
            if counts_by_day.get(d, 0) >= min_sessions:
                temp += 1; best = max(best, temp)
            else:
                temp = 0
        recent = [counts_by_day.get(today - timedelta(days=i), 0) for i in range(7)]
        consistency = safe_div(len([x for x in recent if x >= min_sessions]), 7) * 100.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Streak", f"{cur_streak} days")
        with c2:
            st.metric("Best Streak", f"{best} days")
        with c3:
            st.metric("Weekly Consistency", f"{consistency:.0f}%")

# === HEADER + ROUTER ===
def main_header_and_router():
    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Ensure Indexes"):
        ensure_indexes()
    export_sessions_csv(st.session_state.user if st.session_state.user else "user")

    users = get_all_users()
    if not users:
        add_user("prashanth")
        users = ["prashanth"]
    if st.session_state.user not in users:
        st.session_state.user = users[0]

    c1, c2, c3 = st.columns([2,3,2])
    with c1:
        idx = users.index(st.session_state.user) if st.session_state.user in users else 0
        sel = st.selectbox("User", users, index=idx, key="user_select")
        if sel != st.session_state.user:
            st.session_state.user = sel
            st.rerun()
    with c2:
        pages = ["🎯 Focus Timer","📅 Weekly Planner","📊 Analytics & Review","🧾 Journal"]
        st.session_state.page = st.selectbox("Navigate", pages,
                                             index=pages.index(st.session_state.page) if "page" in st.session_state and st.session_state.page in pages else 0)
    with c3:
        with st.expander("Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add") and u:
                if add_user(u.strip()):
                    st.session_state.user = u.strip()
                    st.success("User added")
                    st.rerun()
                else:
                    st.warning("User already exists")

    st.divider()
    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer()
    elif page == "📅 Weekly Planner":
        render_weekly_planner()
    elif page == "📊 Analytics & Review":
        render_analytics_review()
    elif page == "🧾 Journal":
        render_journal()

if __name__ == "__main__":
    # initialize session defaults
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.user = None
        st.session_state.page = "🎯 Focus Timer"
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""
        st.session_state.planning_week_date = now_ist().date()
        st.session_state.review_week_date = now_ist().date()
    main_header_and_router()
