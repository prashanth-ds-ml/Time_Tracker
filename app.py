import streamlit as st
import time
import os
from datetime import datetime, timedelta, date
import pandas as pd
import pytz
import plotly.express as px
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

# === HELPERS: TIME / WEEK ===
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> Tuple[date, date]:
    # Monday start (ISO), Sunday end
    weekday = d.weekday()  # 0=Mon
    start = d - timedelta(days=weekday)
    end = start + timedelta(days=6)
    return start, end

def week_day_counts(week_start: date) -> Tuple[int, int]:
    """Return (#weekdays, #weekend_days) for the given Monday-start week."""
    wd = 0
    we = 0
    for i in range(7):
        day = week_start + timedelta(days=i)
        if day.weekday() < 5:
            wd += 1
        else:
            we += 1
    return wd, we

# ---- Analytics helpers ----
def safe_div(n, d, default=0.0):
    try:
        if d is None or d == 0:
            return default
        return float(n) / float(d)
    except Exception:
        return default

def pct_or_dash(n, d, decimals=0):
    """Return percent string or '—' if denominator invalid."""
    if d is None or d <= 0:
        return "—"
    pct = 100.0 * safe_div(n, d, default=0.0)
    fmt = f"{{:.{decimals}f}}%"
    return fmt.format(pct)

def gini_from_counts(counts):
    """Gini coefficient (0..1) from a list/Series of non-negative counts."""
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
    """Normalized entropy (0..1). 0=one bucket dominates, 1=uniform."""
    arr = [c for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1:
        return 0.0
    s = float(sum(arr))
    H = -sum((c/s) * math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

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
    # Backward-compat columns for legacy rows
    if 'goal_id' not in df.columns:
        df['goal_id'] = None
    if 'category' not in df.columns:
        df['category'] = ''
    if 'pomodoro_type' not in df.columns:
        df['pomodoro_type'] = 'Work'
    return df

@st.cache_data(ttl=300)
def get_user_data(username: str) -> pd.DataFrame:
    return get_user_sessions(username)

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

# === GOALS ===
def upsert_goal(username: str, title: str, priority_weight: int, goal_type: str, status: str = "New", target_poms: int = 0) -> str:
    # Generate deterministic goal id per user+title
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

# === WEEKLY PLAN ===
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
    # dedupe goals and keep only keys present in allocations
    goals_unique = sorted(set(goals))
    clean_alloc = {gid: int(allocations.get(gid, 0)) for gid in goals_unique}
    collection_plans.update_one(
        {"_id": plan_id},
        {"$set": {"goals": goals_unique, "allocations": clean_alloc, "updated_at": datetime.utcnow()}}
    )

# === LOCKING MECHANISM ===
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
    # increment goal progress if applicable (work only)
    if (not is_break) and goal_id:
        collection_goals.update_one({"_id": goal_id}, {"$inc": {"poms_completed": 1}, "$set": {"updated_at": datetime.utcnow()}})
    get_user_sessions.clear()

# === DAILY TARGETS ===
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
    st.markdown("## 🎯 Daily Target Planner")
    current_target = get_daily_target(st.session_state.user)
    if df.empty:
        suggested_target, phase_name, _ = 1, "🌱 Building", ""
    else:
        df_work = df[df["pomodoro_type"]=="Work"]
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        suggested_target, phase_name, _ = get_adaptive_goal(active_days)
    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown("### 📋 Set Your Target")
        if current_target is not None:
            st.info(f"✅ Today's target: **{current_target} Pomodoros**")
            with st.expander("🔄 Change Today's Target"):
                new_target = st.number_input("New target", 1, 12, value=int(current_target))
                if st.button("💾 Update Target"):
                    save_daily_target(int(new_target), st.session_state.user)
                    st.success("🎯 Target updated!")
                    st.rerun()
        else:
            st.markdown(f"💡 **Suggested:** {suggested_target} Pomodoros ({phase_name})")
            target_input = st.number_input("How many Pomodoros today?", 1, 12, value=int(suggested_target))
            if st.button("🎯 Set Daily Target", use_container_width=True):
                save_daily_target(int(target_input), st.session_state.user)
                st.success("✅ Daily target set!")
                st.rerun()
    with col2:
        if current_target is not None:
            pct = min(100.0, (today_progress / max(1,int(current_target))) * 100)
            st.progress(pct/100.0, text=f"🎯 {pct:.0f}% Complete")
        else:
            st.info("Set a target to unlock enhanced tracking.")

# === SESSION STATE ===
def init_session_state():
    defaults = {
        "start_time": None,
        "is_break": False,
        "task": "",
        "user": None,
        "page": "🎯 Focus Timer",
        "active_goal_id": None,
        "active_goal_title": "",
        "planning_week_date": now_ist().date(),
        "review_week_date": now_ist().date(),
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# === OPTIONAL: ADMIN / UTIL ===
def ensure_indexes():
    try:
        collection_logs.create_index([("user",1),("type",1),("date",1)], name="user_type_date")
        collection_logs.create_index([("goal_id",1)], name="goal_id")
        collection_goals.create_index([("user",1),("title",1)], unique=True, name="user_title_unique")
        collection_plans.create_index([("user",1),("week_start",1)], name="user_week")
        collection_reflections.create_index([("user",1),("date",1)], name="user_date")
        st.success("Indexes ensured/created (if not present).")
    except Exception as e:
        st.warning(f"Index creation notice: {e}")

def export_sessions_csv(user: str):
    df = get_user_sessions(user)
    if df.empty:
        st.info("No sessions to export.")
        return
    out = df.sort_values("date")
    st.download_button("⬇️ Export Sessions (CSV)", out.to_csv(index=False).encode("utf-8"), file_name=f"{user}_sessions.csv", mime="text/csv")

# === WEEKLY PLANNER PAGE ===
def render_weekly_planner():
    st.header("📅 Weekly Goal Planner")

    user = st.session_state.user
    # Week picker
    pick_date = st.date_input("📆 Plan for week of", value=st.session_state.planning_week_date)
    # normalize to Monday
    week_start, week_end = week_bounds_ist(pick_date)
    if pick_date != st.session_state.planning_week_date:
        st.session_state.planning_week_date = pick_date
        st.rerun()

    settings = get_user_settings(user)
    plan = get_or_create_weekly_plan(user, week_start)

    colA, colB, colC = st.columns(3)
    with colA:
        wp = st.number_input("Avg Pomodoros / Weekday", 0, 12, value=settings["weekday_poms"])
    with colB:
        we = st.number_input("Avg Pomodoros / Weekend Day", 0, 12, value=settings["weekend_poms"])
    with colC:
        wd_count, we_count = week_day_counts(week_start)
        total = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we}, weekdays=wd_count, weekend_days=we_count)
        st.metric(f"Weekly Capacity ({week_start} → {week_end})", f"{total}")
        if (wp != settings["weekday_poms"]) or (we != settings["weekend_poms"]):
            if st.button("💾 Save Capacity", use_container_width=True):
                users_collection.update_one({"username": user}, {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
                get_user_settings.clear()
                st.success("Updated user defaults")
                st.rerun()

    st.divider()

    st.subheader("🎯 Define Priorities (Top 3–4)")
    with st.expander("➕ Add or Update Goal", expanded=False):
        g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
        g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        g_weight = st.select_slider("Priority Weight", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
        g_status = st.selectbox("Status", ["New","In Progress","Completed","On Hold","Archived"], index=0)
        if st.button("💾 Save Goal"):
            if g_title.strip():
                upsert_goal(user, g_title.strip(), int(g_weight), g_type, g_status)
                fetch_goals.clear()
                st.success("Saved goal")
                st.rerun()
            else:
                st.warning("Please provide a title")

    goals_df = fetch_goals(user, statuses=["New","In Progress"])
    if goals_df.empty:
        st.info("Add 3–4 goals above to plan this week.")
        return

    st.markdown("#### Active Goals")
    show = goals_df[["_id","title","goal_type","priority_weight","status"]].rename(columns={"_id":"Goal ID","goal_type":"Type","priority_weight":"Weight"})
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🧮 Auto-Allocate Weekly Pomodoros")
    wd_count, we_count = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(get_user_settings(user), weekdays=wd_count, weekend_days=we_count)
    weight_map = {row["_id"]: int(row["priority_weight"]) for _, row in goals_df.iterrows()}
    auto = proportional_allocation(total_poms, weight_map)
    st.caption("Adjust numbers to fine-tune allocations (sum preserved).")

    # if plan is empty, offer to copy last week's plan
    prev_start = week_start - timedelta(days=7)
    prev_plan = collection_plans.find_one({"_id": f"{user}|{prev_start.isoformat()}"})

    # Prevent accidental overwrite flag
    plan_has_alloc = bool(plan.get("allocations"))
    if plan_has_alloc:
        st.warning("A plan already exists for this week. You can update it below. To replace entirely, tick the overwrite box before saving.")

    if (not plan.get("allocations")) and prev_plan and prev_plan.get("allocations"):
        if st.button(f"📋 Copy last week's plan ({prev_start} → {prev_start+timedelta(days=6)})"):
            save_plan_allocations(plan["_id"], list(prev_plan["allocations"].keys()), prev_plan["allocations"])
            st.success("Copied last week's plan")
            st.rerun()

    edited = {}
    cols = st.columns(min(4, len(auto)) if len(auto) > 0 else 1)
    i = 0
    for _, row in goals_df.iterrows():
        with cols[i % len(cols)]:
            default_val = int(plan.get("allocations", {}).get(row['_id'], auto[row['_id']]))
            val = st.number_input(f"{row['title']}", min_value=0, max_value=total_poms, value=default_val, step=1, key=f"alloc_{row['_id']}")
            edited[row["_id"]] = int(val)
        i += 1

    sum_edit = sum(edited.values())
    if sum_edit != total_poms:
        st.warning(f"Allocations sum to {sum_edit}, not {total_poms}. Click to auto-correct.")
        if st.button("🔁 Normalize to Total"):
            edited = proportional_allocation(total_poms, {gid: max(1, v) for gid, v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.rerun()

    overwrite = st.checkbox("I understand this will overwrite the existing plan for this week", value=False, disabled=not plan_has_alloc)
    btn_label = "📌 Update Weekly Plan" if plan_has_alloc else "📌 Save Weekly Plan"
    disabled_save = plan_has_alloc and (not overwrite)

    if st.button(btn_label, type="primary", disabled=disabled_save):
        save_plan_allocations(plan["_id"], list(edited.keys()), edited)
        st.success("Weekly plan saved!")
        st.rerun()

# === TIMER WIDGET ===
def render_timer_widget(auto_break: bool) -> bool:
    if not st.session_state.start_time:
        return False
    duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
    remaining = int(st.session_state.start_time + duration - time.time())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        session_type = "Break Time" if st.session_state.is_break else f"Working on: {st.session_state.task}"
        st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {session_type}")
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
        progress = 1 - (remaining/duration)
        st.progress(progress)
        st.info("🧘 Take a breather!" if st.session_state.is_break else "💪 Stay focused!")
        time.sleep(1)
        st.rerun()
        return True
    else:
        # session completes
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
        st.balloons(); st.success("🎉 Session complete!")

        # Reset work-state
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""

        # Auto-start a break after WORK, if enabled
        if (not was_break) and auto_break:
            st.toast("☕ Auto-starting a 5-minute break")
            st.session_state.start_time = time.time()
            st.session_state.is_break = True
            st.rerun()
        return True

# === FOCUS TIMER PAGE ===
def render_focus_timer():
    st.header("🎯 Focus Timer")

    # user settings (auto-break + custom category list)
    settings = get_user_settings(st.session_state.user)
    colset1, colset2 = st.columns([1, 3])
    with colset1:
        auto_break_ui = st.toggle("Auto-start break", value=settings.get("auto_break", True), help="Start a 5m break automatically after each 25m work session")
        if auto_break_ui != settings.get("auto_break", True):
            users_collection.update_one({"username": st.session_state.user}, {"$set": {"auto_break": bool(auto_break_ui)}})
            get_user_settings.clear()

    # if a timer is active, show the widget and return
    if render_timer_widget(auto_break=get_user_settings(st.session_state.user).get("auto_break", True)):
        return

    user = st.session_state.user
    # Focus Timer always uses the *current week* for locking/context
    plan = get_or_create_weekly_plan(user, now_ist().date())

    # Daily Target Planner (top)
    df_all = get_user_sessions(user)
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df_all)
    render_daily_target_planner(df_all, today_progress)
    st.divider()

    # Mode toggle: Weekly Goal vs Custom (Unplanned)
    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True)

    if mode == "Weekly Goal":
        active_goal_ids = plan.get("goals", [])
        if not active_goal_ids:
            st.warning("No weekly plan saved yet. Please create allocations in the **Weekly Planner** page.")
        goals_df = fetch_goals(user, statuses=["New","In Progress"])
        goals_df = goals_df[goals_df["_id"].isin(active_goal_ids)] if not goals_df.empty else goals_df

        locked = set(locked_goals_for_user_plan(user, plan))
        if locked:
            st.warning("⚖️ Balanced Focus: Top goals temporarily locked. Spend a minimum on other goals to unlock.")

        choices = []
        for _, r in goals_df.iterrows():
            title = r['title']
            gid = r['_id']
            disabled = gid in locked
            alloc = plan.get('allocations', {}).get(gid, 0)
            label = f"{title}  ·  plan:{alloc}"
            choices.append((label, gid, disabled))

        c1, c2 = st.columns([1,2])
        with c1:
            options_labels = [lab + ("  🔒" if dis else "") for (lab,_,dis) in choices] or ["(no goals)"]
            selected_idx = st.selectbox("Weekly Goal", options=range(len(options_labels)), format_func=lambda i: options_labels[i], disabled=len(choices)==0)
            selected_gid = choices[selected_idx][1] if choices else None
            selected_title = choices[selected_idx][0].split('  ·')[0] if choices else ""
        with c2:
            task = st.text_input("Task (micro-task)", placeholder="e.g., Revise Unit-2 notes")

        st.session_state.active_goal_id = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colw, colb = st.columns(2)
        with colw:
            disabled = (not task.strip()) or (selected_gid in locked if selected_gid else False) or (len(choices)==0)
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
            if disabled and len(choices)>0 and selected_gid in locked:
                st.caption("This goal is locked for balance. Switch goal for now.")
        with colb:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.rerun()
    else:
        # Custom (Unplanned) category flow (no goal_id)
        current_cats = get_user_settings(user).get("custom_categories", ["Learning","Projects","Research","Planning"])
        cat_options = current_cats + ["+ Add New"]
        selected = st.selectbox("📂 Custom (Unplanned) Category", cat_options)
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add Category"):
                if new_cat not in current_cats:
                    users_collection.update_one({"username": user}, {"$addToSet": {"custom_categories": new_cat}})
                    get_user_settings.clear()
                    st.success("Category added!")
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
        st.divider(); st.subheader("📊 Today's Summary")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("Work Sessions", len(work_today))
        with col2:
            st.metric("Focus Minutes", int(work_today['duration'].sum()))
        with col3:
            ratio = safe_div(breaks_today, max(1,len(work_today)))
            label = "⚖️ Well balanced" if 0.3<=ratio<=0.7 else ("🎯 More focus" if ratio>0.7 else "🧘 Take breaks")
            st.metric("Breaks", breaks_today, help=label)
        with col4:
            current_target = get_daily_target(user)
            if current_target:
                pct = (len(work_today)/max(1,int(current_target)))*100
                st.metric("Target Progress", f"{pct:.0f}%")
            else:
                st.metric("Target Progress", "—")

# === JOURNAL (Notes + Reflection combined) ===
def render_journal():
    st.header("🧾 Journal")
    tab1, tab2, tab3 = st.tabs(["Reflection", "Add Note", "Browse Notes"])

    user = st.session_state.user
    today_iso = now_ist().date().isoformat()

    with tab1:
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form", clear_on_submit=True):
            aligned = st.selectbox("Was today aligned with the weekly plan?", ["Yes","Partly","No"])
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

# === ANALYTICS & REVIEW ===
def render_analytics_review():
    st.header("📊 Analytics & Review")

    # Mode control (segmented if available, else radio)
    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0)

    user = st.session_state.user
    df_all = get_user_sessions(user)
    if df_all.empty:
        st.info("No sessions yet. Start a Pomodoro to populate analytics.")
        return

    # Split Work vs Break early
    df_all["date_only"] = df_all["date"].dt.date
    df_work = df_all[df_all["pomodoro_type"] == "Work"].copy()
    df_break = df_all[df_all["pomodoro_type"] == "Break"].copy()

    if mode == "Week Review":
        # ---- Week selection & plan ----
        pick_date = st.date_input("📆 Review week of", value=st.session_state.review_week_date)
        if pick_date != st.session_state.review_week_date:
            st.session_state.review_week_date = pick_date
            st.rerun()
        week_start, week_end = week_bounds_ist(pick_date)
        plan = get_or_create_weekly_plan(user, week_start)
        planned_alloc = plan.get("allocations", {}) or {}
        total_planned = int(sum(planned_alloc.values())) if planned_alloc else 0

        # Week slices
        mask_week = (df_all["date_only"] >= week_start) & (df_all["date_only"] <= week_end)
        dfw = df_work[mask_week].copy()
        dfb = df_break[mask_week].copy()

        # Goal vs Custom split
        work_goal = dfw[dfw["goal_id"].notna()].copy()
        work_custom = dfw[dfw["goal_id"].isna()].copy()

        # Deep-work % (duration >= 23)
        deep = len(dfw[dfw["duration"] >= 23])

        # Balance metrics across goals
        goal_counts = work_goal.groupby("goal_id").size().values.tolist()

        # Percent cards (N/A-safe)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Plan Adherence", pct_or_dash(len(work_goal), total_planned))
        with c2:
            st.metric("Capacity Utilization", pct_or_dash(len(dfw), total_planned))
        with c3:
            st.metric("Deep-work %", pct_or_dash(deep, len(dfw)))
        with c4:
            st.metric("Balance (Entropy)", f"{entropy_norm_from_counts(goal_counts):.2f}")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Gini (Goals)", f"{gini_from_counts(goal_counts):.2f}")
        with c6:
            st.metric("Custom (Unplanned) Share", pct_or_dash(len(work_custom), len(dfw)))
        with c7:
            expected_breaks = len(dfw)
            skip = max(0, expected_breaks - len(dfb))
            st.metric("Break Skip", pct_or_dash(skip, expected_breaks))
        with c8:
            extend = max(0, len(dfb) - expected_breaks)
            st.metric("Break Extend", pct_or_dash(extend, expected_breaks))

        # Extra rhythm/discipline readouts
        st.caption(
            f"Avg Break = {(dfb['duration'].mean() if len(dfb) else 0.0):.1f} min • "
            f"Capacity = {total_planned if total_planned>0 else '—'} planned slots "
            f"(weekday {get_user_settings(user)['weekday_poms']} / weekend {get_user_settings(user)['weekend_poms']})"
        )

        # Discipline & Rhythm block
        st.subheader("Discipline & Rhythm")
        # Switching-Cost Index: fraction of transitions that change goal/category
        dfw_sorted = dfw.sort_values(["date", "time"])
        switches = 0
        runs = 0
        prev_key = None
        for _, r in dfw_sorted.iterrows():
            key = r["goal_id"] if pd.notna(r["goal_id"]) else f"CAT::{r.get('category','')}"
            if prev_key is not None:
                if key != prev_key:
                    switches += 1
            prev_key = key
            runs += 1
        switch_idx = safe_div(switches, max(1, runs - 1))
        # Start-time sigma (minutes)
        def time_to_minutes(tstr):
            try:
                dt = datetime.strptime(tstr, "%I:%M %p")
                return dt.hour*60 + dt.minute
            except Exception:
                return None
        starts = [time_to_minutes(x) for x in dfw["time"].tolist() if isinstance(x, str)]
        starts = [s for s in starts if s is not None]
        start_sigma = (pd.Series(starts).std() if len(starts) >= 2 else None)
        # Work/Break ratio
        wb_ratio = safe_div(len(dfw), max(1, len(dfb)))
        # Chronotype (peak hour)
        if len(starts) > 0:
            hours = [s//60 for s in starts]
            peak_hour = pd.Series(hours).mode().iloc[0]
            ampm = "AM" if peak_hour < 12 else "PM"
            ph_disp = f"{(peak_hour if 1 <= peak_hour <= 12 else (12 if peak_hour%12==0 else peak_hour%12))}{ampm}"
        else:
            ph_disp = "—"
        # Task granularity: median sessions per task this week
        if not dfw.empty:
            med_sessions_per_task = dfw.groupby("task").size().median()
        else:
            med_sessions_per_task = None

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Switching-Cost Index", f"{switch_idx*100:.0f}%")
        with d2:
            st.metric("Start-time σ", f"{start_sigma:.0f} min" if start_sigma is not None else "—")
        with d3:
            st.metric("Work/Break Ratio", f"{wb_ratio:.2f}")
        with d4:
            st.metric("Chronotype Window", ph_disp)
        st.caption(f"Task Granularity (median sessions per task): {med_sessions_per_task:.1f}" if med_sessions_per_task is not None else "Task Granularity: —")

        # Duration sanity
        if not dfw.empty:
            off_blocks = len(dfw[(dfw["duration"] < 20) | (dfw["duration"] > 30)])
            st.caption("Durations look clean" if off_blocks == 0 else f"⚠️ {off_blocks} sessions deviate from 25±5 min")

        st.divider()

        # ---- Per-Goal Adherence chart/table ----
        st.subheader("Per-Goal Adherence")

        def title_of(gid):
            if gid is None:
                return "Custom (Unplanned)"
            doc = collection_goals.find_one({"_id": gid})
            return doc["title"] if doc else "(missing)"

        planned_df = pd.DataFrame(
            [{"goal_id": gid, "planned": int(v), "title": title_of(gid)} for gid, v in planned_alloc.items()]
        )
        actual_df = (
            dfw.groupby(dfw["goal_id"])
               .size()
               .rename("actual")
               .reset_index()
        )
        actual_df["title"] = actual_df["goal_id"].apply(title_of)

        merged = pd.merge(planned_df, actual_df, on=["goal_id", "title"], how="outer").fillna(0)
        merged["planned"] = merged["planned"].astype(int)
        merged["actual"] = merged["actual"].astype(int)
        if merged["goal_id"].isna().any():
            merged.loc[merged["goal_id"].isna(), "goal_id"] = None

        cA, cB = st.columns([3, 2])
        with cA:
            if not merged.empty:
                fig = px.bar(
                    merged.sort_values("planned", ascending=False),
                    x="title", y=["planned", "actual"],
                    barmode="group",
                    title="Planned vs Actual Pomodoros",
                )
                fig.update_layout(height=360, xaxis_title="", legend_title="")
                st.plotly_chart(fig, use_container_width=True)

        with cB:
            total_actual_goals = int(work_goal.shape[0])
            carry = max(0, total_planned - total_actual_goals)
            st.metric("Planned (week)", total_planned if total_planned>0 else 0)
            st.metric("Actual (goals)", total_actual_goals)
            st.metric("Carryover", carry)
            st.metric("Carryover Rate", pct_or_dash(carry, total_planned))

        # Optional diagnostics (hidden by default)
        show_diag = st.toggle("Show diagnostics", value=False)
        if show_diag:
            missing_link_pct = pct_or_dash(
                len(dfw[dfw["goal_id"].isna() & (len(planned_alloc) > 0)]),
                len(dfw)
            )
            st.caption(f"Missing Link (no goal_id in planned week): {missing_link_pct}")

        st.divider()

        # ---- Run-rate vs Expected (cumulative) ----
        if total_planned > 0:
            days = pd.date_range(
                start=pd.to_datetime(week_start),
                end=pd.to_datetime(min(week_end, now_ist().date()))
            )
            dfw_goal = dfw[dfw["goal_id"].notna()].copy()
            dfw_goal["date_only"] = dfw_goal["date"].dt.date

            actual_cum, exp_cum = [], []
            for i, ts in enumerate(days):
                cutoff = ts.date()
                actual_to_d = int((dfw_goal["date_only"] <= cutoff).sum())
                expected_to_d = int(round(total_planned * ((i + 1) / len(days))))
                actual_cum.append(actual_to_d)
                exp_cum.append(expected_to_d)

            rr_df = pd.DataFrame({
                "day": [ts.strftime("%a %d") for ts in days],
                "Expected": exp_cum,
                "Actual": actual_cum
            })
            fig_rr = px.line(rr_df, x="day", y=["Expected", "Actual"], markers=True,
                             title="Run-Rate vs Expected (Goals only)")
            fig_rr.update_layout(height=330, legend_title="")
            st.plotly_chart(fig_rr, use_container_width=True)

        # ---- Close Out & Rollover (only if something to close) ----
        has_planned = bool(planned_alloc)
        has_goal_actuals = (work_goal.shape[0] > 0)
        if has_planned or has_goal_actuals:
            st.divider()
            st.subheader("Close Out & Rollover")
            if not merged.empty:
                for _, row in merged.sort_values("title").iterrows():
                    gid = row["goal_id"]
                    if gid in (None, "NONE"):
                        continue  # skip custom for closeout
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    with col1:
                        st.write(f"**{row['title']}**")
                    with col2:
                        status = st.selectbox(
                            "Status",
                            ["Completed", "Rollover", "On Hold", "Archived", "In Progress"],
                            index=4,
                            key=f"close_{gid}_{week_start}",
                        )
                    with col3:
                        carry = max(0, int(row["planned"]) - int(row["actual"]))
                        carry = st.number_input(
                            "Carry fwd poms",
                            0, 200, value=carry,
                            key=f"carry_{gid}_{week_start}",
                        )
                    with col4:
                        if st.button("✅ Apply", key=f"apply_{gid}_{week_start}"):
                            new_status = (
                                "Completed" if status == "Completed" else
                                "On Hold" if status == "On Hold" else
                                "Archived" if status == "Archived" else
                                "In Progress"
                            )
                            collection_goals.update_one({"_id": gid}, {"$set": {"status": new_status}})

                            if status == "Rollover" and carry > 0:
                                next_start = week_start + timedelta(days=7)
                                next_plan = get_or_create_weekly_plan(user, next_start)
                                next_alloc = next_plan.get("allocations", {}) or {}
                                next_goals = set(next_plan.get("goals", []))
                                next_goals.add(gid)
                                next_alloc[gid] = next_alloc.get(gid, 0) + int(carry)
                                collection_plans.update_one(
                                    {"_id": next_plan["_id"]},
                                    {"$set": {
                                        "goals": list(next_goals),
                                        "allocations": next_alloc,
                                        "updated_at": datetime.utcnow()
                                    }}
                                )
                            st.success("Updated")
        else:
            st.info("Nothing to close this week yet. Log some goal-linked sessions first.")

    else:
        # ---- Trends mode ----
        today = now_ist().date()

        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Total Sessions", len(df_work))
        with col2:
            st.metric("⏱️ Total Hours", int(df_work["duration"].sum() // 60))
        with col3:
            st.metric("📅 Active Days", int(df_work.groupby("date_only").size().shape[0]))
        with col4:
            avg_daily = df_work.groupby("date_only").size().mean() if len(df_work) else 0
            st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

        st.divider()
        st.subheader("📈 Daily Performance (Last 30 Days)")
        daily_data = []
        for i in range(30):
            d = today - timedelta(days=29 - i)
            dwork = df_work[df_work["date_only"] == d]
            daily_data.append({
                "date": d.strftime("%m/%d"),
                "minutes": int(dwork["duration"].sum()),
            })
        daily_df = pd.DataFrame(daily_data)
        if daily_df["minutes"].sum() > 0:
            fig = px.bar(daily_df, x="date", y="minutes", title="Daily Focus Minutes",
                         color="minutes", color_continuous_scale="Blues")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🎯 Category Deep Dive")

        time_filter = st.selectbox("📅 Time Period", ["Last 7 days", "Last 30 days", "All time"], index=1)
        if time_filter == "Last 7 days":
            cutoff = today - timedelta(days=7)
            fw = df_work[df_work["date_only"] >= cutoff]
        elif time_filter == "Last 30 days":
            cutoff = today - timedelta(days=30)
            fw = df_work[df_work["date_only"] >= cutoff]
        else:
            fw = df_work

        if fw.empty:
            st.info(f"No data available for {time_filter.lower()}.")
            return

        cat_stats = fw.groupby("category").agg(duration=("duration", "sum"),
                                               sessions=("duration", "count")).sort_values("duration", ascending=False)
        colA, colB = st.columns([3, 2])
        with colA:
            total_time = cat_stats["duration"].sum()
            fig_donut = px.pie(
                values=cat_stats["duration"],
                names=cat_stats.index,
                title=f"📊 Time Distribution by Category ({time_filter})",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            total_hours = int(total_time) // 60
            total_mins = int(total_time) % 60
            center_text = f"{total_hours}h {total_mins}m" if total_hours > 0 else f"{total_mins}m"
            fig_donut.add_annotation(text=f"<b>Total</b><br>{center_text}", x=0.5, y=0.5, showarrow=False)
            fig_donut.update_layout(height=400, showlegend=True, title_x=0.5)
            st.plotly_chart(fig_donut, use_container_width=True)
        with colB:
            st.markdown("#### 📈 Category Performance")
            view = cat_stats.copy()
            view["Time"] = view["duration"].apply(lambda m: f"{int(m//60)}h {int(m%60)}m" if m >= 60 else f"{int(m)}m")
            view["Avg/Session"] = (view["duration"] / view["sessions"]).round(1).astype(str) + "m"
            st.dataframe(
                view[["Time", "sessions", "Avg/Session"]],
                use_container_width=True, hide_index=False,
                height=min(len(view) * 35 + 38, 300)
            )

        st.subheader("🎯 Task Performance")
        tstats = fw.groupby(["category", "task"]).agg(total_minutes=("duration", "sum"),
                                                      sessions=("duration", "count")).reset_index()
        tstats = tstats.sort_values("total_minutes", ascending=False)
        colC, colD = st.columns([3, 2])
        with colC:
            top_tasks = tstats.head(12)
            if not top_tasks.empty:
                fig_tasks = px.bar(
                    top_tasks, x="total_minutes", y="task", color="category",
                    title=f"Top Tasks by Time Investment ({time_filter})",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_tasks.update_layout(height=max(400, len(top_tasks) * 30),
                                        yaxis={"categoryorder": "total ascending"},
                                        title_x=0.5, showlegend=True)
                st.plotly_chart(fig_tasks, use_container_width=True)
        with colD:
            st.markdown("#### 💡 Insights")
            if not tstats.empty:
                total_time = tstats["total_minutes"].sum()
                top = tstats.iloc[0]
                share = safe_div(top["total_minutes"], total_time) * 100
                if share > 50:
                    st.warning("⚖️ One task dominates your time. Consider rebalancing.")
                elif share > 25:
                    st.info("🎯 Clear primary task focus this period.")
                else:
                    st.success("✅ Time is well distributed across tasks.")

        # Consistency
        st.divider()
        st.subheader("🔥 Consistency Tracking")
        counts_by_day = df_work.groupby("date_only").size().to_dict()
        active_days = len(counts_by_day)
        min_sessions = 1 if active_days <= 12 else 2

        # Current streak
        cur_streak = 0
        for i in range(365):
            d = today - timedelta(days=i)
            if counts_by_day.get(d, 0) >= min_sessions:
                cur_streak += 1
            else:
                break

        # Best streak
        best, temp = 0, 0
        for i in range(365):
            d = today - timedelta(days=i)
            if counts_by_day.get(d, 0) >= min_sessions:
                temp += 1
                best = max(best, temp)
            else:
                temp = 0

        recent = [counts_by_day.get(today - timedelta(days=i), 0) for i in range(7)]
        consistency = safe_div(len([x for x in recent if x >= min_sessions]), 7) * 100.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🔥 Current Streak", f"{cur_streak} days")
        with c2:
            st.metric("🏆 Best Streak", f"{best} days")
        with c3:
            st.metric("📊 Weekly Consistency", f"{consistency:.0f}%")

# === HEADER + ROUTER ===
def main_header_and_router():
    st.sidebar.markdown("### ⚙️ Admin")
    if st.sidebar.button("Initialize Mongo Indexes"):
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
        sel = st.selectbox("👤 User", users, index=idx, key="user_select")
        if sel != st.session_state.user:
            st.session_state.user = sel
            st.rerun()
    with c2:
        pages = [
            "🎯 Focus Timer",
            "📅 Weekly Planner",
            "📊 Analytics & Review",
            "🧾 Journal",
        ]
        st.session_state.page = st.selectbox("📍 Navigate", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    with c3:
        with st.expander("➕ Add User"):
            u = st.text_input("Username", key="new_user_input")
            if st.button("Add", key="add_user_btn") and u:
                if add_user(u.strip()):
                    st.session_state.user = u.strip()
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
