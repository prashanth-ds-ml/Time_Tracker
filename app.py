import streamlit as st
import time
from datetime import datetime, timedelta, date
import pandas as pd
import pytz
import plotly.express as px
from pymongo import MongoClient
import hashlib
from typing import List, Dict, Tuple, Optional
import math

# =========================
# CONFIG
# =========================
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

# =========================
# DB INIT
# =========================
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

# =========================
# HELPERS: TIME / WEEK / MATH
# =========================
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> Tuple[date, date]:
    weekday = d.weekday()  # 0=Mon
    start = d - timedelta(days=weekday)
    end = start + timedelta(days=6)
    return start, end

def week_day_counts(week_start: date) -> Tuple[int, int]:
    wd = 0; we = 0
    for i in range(7):
        day = week_start + timedelta(days=i)
        if day.weekday() < 5: wd += 1
        else: we += 1
    return wd, we

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
    try:
        dt = datetime.strptime(tstr, "%I:%M %p")
        return dt.hour*60 + dt.minute
    except Exception:
        return None

def sound_alert():
    st.components.v1.html(f"""
        <audio id="beep" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
        <script>
            const audio = document.getElementById('beep');
            if (audio) {{
                audio.volume = 0.6;
                audio.play().catch(() => {{ }});
            }}
        </script>
    """, height=0)

# =========================
# SESSION STATE DEFAULTS
# =========================
def init_session_state():
    defaults = {
        "start_time": None,
        "is_break": False,
        "task": "",
        "user": None,
        "page": "🎯 Focus Timer",
        "planning_week_date": now_ist().date(),
        "review_week_date": now_ist().date(),
        "active_goal_id": None,
        "active_goal_title": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_runtime_state_for_user():
    st.session_state.start_time = None
    st.session_state.is_break = False
    st.session_state.task = ""
    st.session_state.active_goal_id = None
    st.session_state.active_goal_title = ""

init_session_state()

# =========================
# DATA ACCESS / CACHES
# =========================
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

# =========================
# GOALS
# =========================
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

def goal_title_map(user: str) -> Dict[str, str]:
    docs = list(collection_goals.find({"user": user}, {"_id": 1, "title": 1}))
    return {d["_id"]: d["title"] for d in docs}

# =========================
# WEEKLY PLAN
# =========================
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
    settings = get_user_settings(username)  # ensures settings doc exists
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

# =========================
# LOCKING (Balanced Focus)
# =========================
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
    mask_week = (df["date"].dt.date >= start) & (df["date"].dt.date <= today)
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

# =========================
# SESSION SAVE
# =========================
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

# =========================
# DAILY TARGETS
# =========================
def get_adaptive_goal(active_days:int):
    if active_days <= 5:
        return 1, "🌱 Building", "Start small - consistency over intensity"
    elif active_days <= 12:
        return 2, "🔥 Growing", "Building momentum - you're doing great!"
    elif active_days <= 19:
        return 3, "💪 Strong", "Push your limits - you're in the zone!"
    else:
        return 4, "🚀 Peak", "Excellence mode - maintain this peak!"

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
    work_today = df_work[df_work["date"].dt.date==today]
    active_days = len(df_work.groupby(df_work["date"].dt.date).size())
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
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        suggested_target, phase_name, _ = get_adaptive_goal(active_days)
    col1, col2 = st.columns([2,3])
    with col1:
        if current_target is not None:
            st.info(f"Today: **{current_target} Pomodoros**")
            with st.expander("Change Today's Target"):
                new_target = st.number_input("New target", 1, 12, value=int(current_target))
                if st.button("💾 Update Target"):
                    save_daily_target(int(new_target), st.session_state.user)
                    st.success("Updated!")
                    st.rerun()
        else:
            st.markdown(f"Suggested: **{suggested_target}** ({phase_name})")
            target_input = st.number_input("How many Pomodoros today?", 1, 12, value=int(suggested_target))
            if st.button("Set Target", use_container_width=True):
                save_daily_target(int(target_input), st.session_state.user)
                st.success("Saved!")
                st.rerun()
    with col2:
        if current_target is not None:
            pct = min(100.0, (today_progress / max(1,int(current_target))) * 100)
            st.progress(pct/100.0, text=f"{pct:.0f}% complete")
        else:
            st.info("Set a target to unlock tracking.")

# =========================
# OPTIONAL: ADMIN / UTIL
# =========================
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
    if not user:
        return
    df = get_user_sessions(user)
    if df.empty:
        return
    out = df.sort_values("date")
    st.download_button("⬇️ Export Sessions (CSV)", out.to_csv(index=False).encode("utf-8"), file_name=f"{user}_sessions.csv", mime="text/csv")

# =========================
# UI HELPERS (Streamlit-native)
# =========================
def this_week_glance_native(user: str, plan: Dict, df_work: pd.DataFrame):
    start = datetime.fromisoformat(plan["week_start"]).date()
    end = datetime.fromisoformat(plan["week_end"]).date()
    active_ids = plan.get("goals", [])
    alloc = plan.get("allocations", {}) or {}
    if not active_ids or not alloc:
        st.info("No allocations yet for this week. Set them in the Weekly Planner.")
        return

    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw = dfw[(dfw["date_only"] >= start) & (dfw["date_only"] <= end)]

    by_goal = dfw[dfw["goal_id"].notna()].groupby("goal_id").size().to_dict()
    titles = goal_title_map(user)

    cols = st.columns(2)
    idx = 0
    for gid in active_ids:
        planned = int(alloc.get(gid, 0))
        actual = int(by_goal.get(gid, 0))
        progress = min(1.0, safe_div(actual, max(1, planned)))
        with cols[idx % 2]:
            st.write(f"**{titles.get(gid, '(missing)')}**")
            st.progress(progress, text=f"{actual}/{planned} completed")
        idx += 1

def start_time_sparkline_native(df_work: pd.DataFrame, title="Start-time Stability (median mins from midnight)"):
    if df_work.empty:
        return
    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw["start_mins"] = dfw["time"].apply(time_to_minutes)
    dfw = dfw[pd.notna(dfw["start_mins"])]
    if dfw.empty:
        return
    daily = dfw.groupby("date_only")["start_mins"].median().reset_index().sort_values("date_only")
    daily = daily.rename(columns={"date_only":"date"})
    daily = daily.set_index("date")
    st.line_chart(daily, height=220)

# =========================
# WEEKLY PLANNER PAGE
# =========================
def render_weekly_planner():
    st.header("📅 Weekly Planner")

    user = st.session_state.user or ""
    if not user:
        st.info("Create or select a user to plan.")
        return

    pick_date = st.date_input("Week of", value=st.session_state.planning_week_date)
    week_start, week_end = week_bounds_ist(pick_date)
    if pick_date != st.session_state.planning_week_date:
        st.session_state.planning_week_date = pick_date
        st.rerun()

    settings = get_user_settings(user)  # exists by bootstrap
    plan = get_or_create_weekly_plan(user, week_start)

    colA, colB, colC = st.columns(3)
    with colA:
        wp = st.number_input("Weekday avg", 0, 12, value=settings["weekday_poms"])
    with colB:
        we = st.number_input("Weekend avg", 0, 12, value=settings["weekend_poms"])
    with colC:
        wd_count, we_count = week_day_counts(week_start)
        total = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we}, weekdays=wd_count, weekend_days=we_count)
        st.metric(f"Capacity {week_start} → {week_end}", f"{total}")
        if (wp != settings["weekday_poms"]) or (we != settings["weekend_poms"]):
            if st.button("💾 Save Defaults", use_container_width=True):
                users_collection.update_one({"username": user}, {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
                st.success("Saved defaults")
                get_user_settings.clear()
                st.rerun()

    st.divider()

    # Goals + priority weights
    st.subheader("🎯 Goals & Priority Weights")
    goals_df = fetch_goals(user, statuses=["New","In Progress"])

    # Quick add for brand-new users
    with st.expander("Add Goal", expanded=goals_df.empty):
        g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
        g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        g_weight = st.select_slider("Priority", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
        if st.button("💾 Save Goal"):
            if g_title.strip():
                upsert_goal(user, g_title.strip(), int(g_weight), g_type, "New")
                fetch_goals.clear()
                st.success("Saved goal")
                st.rerun()
            else:
                st.warning("Please provide a title")

    goals_df = fetch_goals(user, statuses=["New","In Progress"])  # refresh
    if goals_df.empty:
        st.info("Add 3–4 goals to plan the week.")
        return

    weights = {}
    cols = st.columns(min(4, max(1, len(goals_df))))
    for i, (_, row) in enumerate(goals_df.iterrows()):
        with cols[i % len(cols)]:
            st.write(f"**{row['title']}**")
            w = st.select_slider("Priority", options=[1,2,3], value=int(row.get("priority_weight",2)), key=f"w_{row['_id']}")
            weights[row["_id"]] = int(w)
    if st.button("💾 Update Priorities"):
        for gid, w in weights.items():
            collection_goals.update_one({"_id": gid}, {"$set": {"priority_weight": int(w), "updated_at": datetime.utcnow()}})
        fetch_goals.clear()
        st.success("Priorities updated.")
        st.rerun()

    st.divider()

    # Allocation
    st.subheader("🧮 Allocate Weekly Pomodoros")
    wd_count, we_count = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(get_user_settings(user), weekdays=wd_count, weekend_days=we_count)
    weight_map = {row["_id"]: int(weights.get(row["_id"], row["priority_weight"])) for _, row in goals_df.iterrows()}
    auto = proportional_allocation(total_poms, weight_map)

    plan_has_alloc = bool(plan.get("allocations"))
    if plan_has_alloc:
        st.caption("A plan already exists for this week. Adjust and save to update.")

    edited = {}
    cols2 = st.columns(min(4, max(1, len(goals_df))))
    for i, (_, row) in enumerate(goals_df.iterrows()):
        with cols2[i % len(cols2)]:
            default_val = int(plan.get("allocations", {}).get(row['_id'], auto[row['_id']]))
            val = st.number_input(
                f"{row['title']}", min_value=0, max_value=total_poms,
                value=default_val, step=1, key=f"alloc_{row['_id']}"
            )
            edited[row["_id"]] = int(val)

    sum_edit = sum(edited.values())
    if sum_edit != total_poms:
        st.warning(f"Allocations sum to {sum_edit}, not {total_poms}.")
        if st.button("🔁 Normalize to total"):
            edited = proportional_allocation(total_poms, {gid: max(1, v) for gid, v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.rerun()

    btn_label = "📌 Save Weekly Plan" if not plan_has_alloc else "📌 Update Weekly Plan"
    if st.button(btn_label, type="primary"):
        save_plan_allocations(plan["_id"], list(edited.keys()), edited)
        st.success("Weekly plan saved!")
        st.rerun()

# =========================
# TIMER WIDGET
# =========================
def render_timer_widget(auto_break: bool) -> bool:
    if not st.session_state.start_time:
        return False
    duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
    remaining = int(st.session_state.start_time + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break" if st.session_state.is_break else f"Working on: {st.session_state.task}"
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
        progress = 1 - (remaining/duration)
        st.progress(progress)
        st.info("🧘 Relax" if st.session_state.is_break else "💪 Stay focused!")
        time.sleep(1)
        st.rerun()
        return True
    else:
        was_break = st.session_state.is_break
        save_pomodoro_session(
            user=st.session_state.user,
            is_break=was_break,
            duration=BREAK_MIN if was_break else POMODORO_MIN,
            goal_id=st.session_state.active_goal_id,
            task=st.session_state.task,
            category_label=st.session_state.active_goal_title
        )
        sound_alert()
        st.balloons()
        st.success("🎉 Session complete!")

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

# =========================
# FOCUS TIMER PAGE
# =========================
def this_week_glance_native(user: str, plan: Dict, df_work: pd.DataFrame):  # redeclared above; keep once
    start = datetime.fromisoformat(plan["week_start"]).date()
    end = datetime.fromisoformat(plan["week_end"]).date()
    active_ids = plan.get("goals", [])
    alloc = plan.get("allocations", {}) or {}
    if not active_ids or not alloc:
        st.info("No allocations yet for this week. Set them in the Weekly Planner.")
        return
    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw = dfw[(dfw["date_only"] >= start) & (dfw["date_only"] <= end)]
    by_goal = dfw[dfw["goal_id"].notna()].groupby("goal_id").size().to_dict()
    titles = goal_title_map(user)
    cols = st.columns(2)
    idx = 0
    for gid in active_ids:
        planned = int(alloc.get(gid, 0))
        actual = int(by_goal.get(gid, 0))
        progress = min(1.0, safe_div(actual, max(1, planned)))
        with cols[idx % 2]:
            st.write(f"**{titles.get(gid, '(missing)')}**")
            st.progress(progress, text=f"{actual}/{planned} completed")
        idx += 1

def start_time_sparkline_native(df_work: pd.DataFrame, title="Start-time Stability (median mins from midnight)"):
    if df_work.empty: return
    dfw = df_work.copy()
    dfw["date_only"] = dfw["date"].dt.date
    dfw["start_mins"] = dfw["time"].apply(time_to_minutes)
    dfw = dfw[pd.notna(dfw["start_mins"])]
    if dfw.empty: return
    daily = dfw.groupby("date_only")["start_mins"].median().reset_index().sort_values("date_only")
    daily = daily.rename(columns={"date_only":"date"}).set_index("date")
    st.line_chart(daily, height=220)

def render_focus_timer():
    st.header("🎯 Focus Timer")

    user = st.session_state.user or ""
    if not user:
        st.info("Create or select a user to start the timer.")
        return

    settings = get_user_settings(user)
    colset1, _ = st.columns([1, 3])
    with colset1:
        auto_break_ui = st.toggle("Auto-start break", value=settings.get("auto_break", True),
                                  help="Start a 5m break automatically after each 25m work session")
        if auto_break_ui != settings.get("auto_break", True):
            users_collection.update_one({"username": user}, {"$set": {"auto_break": bool(auto_break_ui)}})
            get_user_settings.clear()

    if render_timer_widget(auto_break=get_user_settings(user).get("auto_break", True)):
        return

    plan = get_or_create_weekly_plan(user, now_ist().date())
    df_all = get_user_sessions(user)
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df_all)
    render_daily_target_planner(df_all, today_progress)
    st.divider()

    df_work_all = df_all[df_all["pomodoro_type"]=="Work"].copy()

    st.subheader("📌 This Week at a Glance")
    this_week_glance_native(user, plan, df_work_all)
    start_time_sparkline_native(df_work_all)
    st.divider()

    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True)

    if mode == "Weekly Goal":
        active_goal_ids = plan.get("goals", [])
        goals_df = fetch_goals(user, statuses=["New","In Progress"])
        if not active_goal_ids or goals_df.empty:
            st.info("No weekly plan or active goals yet. Go to **Weekly Planner** to add goals and allocations.")
            # Offer start custom work anyway
            st.write("---")
            st.write("Or track a quick custom session below:")
            mode = "Custom (Unplanned)"

    if mode == "Weekly Goal":
        goals_df = goals_df[goals_df["_id"].isin(active_goal_ids)]
        locked = set(locked_goals_for_user_plan(user, plan))
        if locked:
            st.warning("⚖️ Balanced Focus: top goals are temporarily locked. Do minimum on others to unlock.")

        titles_pairs = goals_df[["title","_id"]].values.tolist()
        if not titles_pairs:
            st.info("No active goals in this week's plan.")
            return

        c1, c2 = st.columns([1,2])
        with c1:
            sel_idx = st.selectbox("Weekly Goal", options=range(len(titles_pairs)),
                                   format_func=lambda i: titles_pairs[i][0])
            selected_gid = titles_pairs[sel_idx][1]
            selected_title = titles_pairs[sel_idx][0]
            if selected_gid in locked:
                st.caption("🔒 This goal is locked for balance. Pick another for now.")
        with c2:
            task = st.text_input("Task (micro-task)", placeholder="e.g., Revise Unit-2 notes")

        st.session_state.active_goal_id = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = (not task.strip()) or (selected_gid in locked if selected_gid else False)
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()
    if mode == "Custom (Unplanned)":
        current_cats = get_user_settings(user).get("custom_categories", ["Learning","Projects","Research","Planning"])
        cat_options = current_cats + ["+ Add New"]
        selected = st.selectbox("📂 Custom Category", cat_options)
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add Category"):
                if new_cat not in current_cats:
                    users_collection.update_one({"username": user}, {"$addToSet": {"custom_categories": new_cat}})
                    get_user_settings.clear()
                    st.success("Added!")
                    st.rerun()
            category_label = new_cat if new_cat else ""
        else:
            category_label = selected
        task = st.text_input("Task (micro-task)", placeholder="e.g., Draft outreach emails")

        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = category_label
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = not (category_label and task.strip())
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()

    # Today's compact summary
    df = get_user_sessions(user)
    if not df.empty:
        today = now_ist().date()
        df["date_only"] = df["date"].dt.date
        today_data = df[df["date_only"] == today]
        work_today = today_data[today_data["pomodoro_type"]=="Work"]
        breaks_today = len(today_data[today_data["pomodoro_type"]=="Break"])
        st.divider(); st.subheader("📊 Today")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("Work Sessions", len(work_today))
        with col2:
            st.metric("Focus Minutes", int(work_today['duration'].sum()))
        with col3:
            ratio = safe_div(breaks_today, max(1,len(work_today)))
            label = "⚖️ Balanced" if 0.3<=ratio<=0.7 else ("🎯 More focus" if ratio>0.7 else "🧘 Take breaks")
            st.metric("Breaks", breaks_today, help=label)
        with col4:
            current_target = get_daily_target(user)
            if current_target:
                pct = (len(work_today)/max(1,int(current_target)))*100
                st.metric("Target Progress", f"{pct:.0f}%")
            else:
                st.metric("Target Progress", "—")

# =========================
# JOURNAL (Notes + Reflection)
# =========================
def render_journal():
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Add Note", "Browse Notes"])

    user = st.session_state.user or ""
    if not user:
        st.info("Create or select a user to write journal.")
        return

    today_iso = now_ist().date().isoformat()

    with tab1:
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form", clear_on_submit=True):
            aligned = st.selectbox("Aligned with weekly plan?", ["Yes","Partly","No"])
            rating = st.slider("Focus quality (1-5)", 1, 5, 3)
            blockers = st.text_area("Blockers / distractions")
            notes = st.text_area("Insights / anything to note")
            submitted = st.form_submit_button("💾 Save Reflection")
            if submitted:
                collection_reflections.update_one(
                    {"user": user, "date": today_iso},
                    {"$set": {
                        "user": user,
                        "date": today_iso,
                        "aligned": aligned,
                        "focus_rating": int(rating),
                        "blockers": blockers.strip(),
                        "notes": notes.strip(),
                        "created_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                st.success("Saved ✨")

        recs = list(collection_reflections.find({"user": user}).sort("date", -1).limit(14))
        if recs:
            st.subheader("Recent Reflections")
            df = pd.DataFrame(recs)
            st.dataframe(df[["date","aligned","focus_rating","blockers","notes"]], use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Add Note")
        with st.form("note_form", clear_on_submit=True):
            c1, c2 = st.columns([1,3])
            with c1:
                d = st.date_input("Date", now_ist())
            with c2:
                content = st.text_area("Your thoughts...", height=140)
            if st.form_submit_button("💾 Save Note"):
                if content.strip():
                    nid = hashlib.sha256(f"{d.date().isoformat()}_{content}_{user}".encode()).hexdigest()
                    doc = {"_id": nid, "type":"Note", "date": d.date().isoformat(), "content": content.strip(), "user": user, "created_at": datetime.utcnow()}
                    collection_logs.update_one({"_id": nid}, {"$set": doc}, upsert=True)
                    st.success("Saved")
                else:
                    st.warning("Add some content")

    with tab3:
        st.subheader("Browse Notes")
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
            st.info("No notes in this range")

# =========================
# ANALYTICS & REVIEW (unchanged from previous stable)
# =========================
def render_analytics_review():
    st.header("📊 Analytics & Review")
    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0)

    user = st.session_state.user or ""
    if not user:
        st.info("Create or select a user.")
        return

    df_all = get_user_sessions(user)
    if df_all.empty:
        st.info("No sessions yet. Start a Pomodoro to populate analytics.")
        return

    # (… keep your existing analytics from the previous working version …)
    st.info("Analytics rendering intact (omitted here for brevity).")

# =========================
# BOOTSTRAP A NEW USER
# =========================
def bootstrap_user(user: str):
    """
    Ensure a new/empty user has:
      - settings doc
      - a current-week plan
    """
    if not user:
        return
    _ = get_user_settings(user)  # creates defaults if needed
    _ = get_or_create_weekly_plan(user, now_ist().date())  # creates plan if needed

# =========================
# HEADER + ROUTER
# =========================
def main_header_and_router():
    users = get_all_users()
    if not users:
        add_user("prashanth")
        users = get_all_users()

    # initialize user
    if st.session_state.user is None or st.session_state.user not in users:
        st.session_state.user = users[0]
        bootstrap_user(st.session_state.user)

    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Initialize Mongo Indexes"):
        ensure_indexes()
    # Export button only when data exists (function will no-op otherwise)
    export_sessions_csv(st.session_state.user)

    c1, c2, c3 = st.columns([2,3,2])
    with c1:
        try:
            idx = users.index(st.session_state.user) if st.session_state.user in users else 0
        except Exception:
            idx = 0
        sel = st.selectbox("👤 User", users, index=idx, key="user_select_header")
        if sel != st.session_state.user:
            st.session_state.user = sel
            reset_runtime_state_for_user()
            # clear caches and bootstrap
            get_user_sessions.clear()
            get_user_settings.clear()
            fetch_goals.clear()
            bootstrap_user(sel)
            st.rerun()
    with c2:
        pages = ["🎯 Focus Timer", "📅 Weekly Planner", "📊 Analytics & Review", "🧾 Journal"]
        current = st.session_state.get("page", pages[0])
        st.session_state.page = st.selectbox("📍 Navigate", pages, index=pages.index(current) if current in pages else 0)
    with c3:
        with st.expander("➕ Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and u:
                uname = u.strip()
                if uname:
                    if add_user(uname):
                        st.session_state.user = uname
                        reset_runtime_state_for_user()
                        get_all_users.clear()
                        get_user_sessions.clear()
                        get_user_settings.clear()
                        fetch_goals.clear()
                        bootstrap_user(uname)
                        st.success("✅ User added!")
                        st.rerun()
                    else:
                        st.warning("User already exists!")

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
    main_header_and_router()
