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
    wd = 0; we = 0
    for i in range(7):
        day = week_start + timedelta(days=i)
        if day.weekday() < 5: wd += 1
        else: we += 1
    return wd, we

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
    if 'goal_id' not in df.columns: df['goal_id'] = None
    if 'category' not in df.columns: df['category'] = ''
    if 'pomodoro_type' not in df.columns: df['pomodoro_type'] = 'Work'
    if 'work_stream_type' not in df.columns:
        df['work_stream_type'] = df.apply(
            lambda r: ("Goal" if (r.get('pomodoro_type') == "Work" and pd.notna(r.get('goal_id')) and r.get('goal_id') not in [None, ""])
                       else ("Custom" if r.get('pomodoro_type') == "Work" else "")),
            axis=1
        )
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
            "weekend_poms": 5
        })
        doc = users_collection.find_one({"username": username})
    return {
        "weekday_poms": int(doc.get("weekday_poms", 3)),
        "weekend_poms": int(doc.get("weekend_poms", 5)),
    }

@st.cache_data(ttl=60)
def get_all_users() -> List[str]:
    return [u["username"] for u in users_collection.find({}, {"_id": 0, "username": 1})]

def add_user(username: str) -> bool:
    if not users_collection.find_one({"username": username}):
        users_collection.insert_one({"username": username, "created_at": datetime.utcnow(), "weekday_poms": 3, "weekend_poms": 5})
        get_all_users.clear()
        return True
    return False

# === GOALS ===
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
    start, end = week_bounds_ist(d)
    pid = f"{username}|{start.isoformat()}"
    plan = collection_plans.find_one({"_id": pid})
    if plan:
        return plan
    settings = get_user_settings(username)
    wd, we = week_day_counts(start)
    total_poms = compute_weekly_capacity(settings, weekdays=wd, weekend_days=we)
    doc = {
        "_id": pid,
        "user": username,
        "week_start": start.isoformat(),
        "week_end": end.isoformat(),
        "total_poms": total_poms,
        "goals": [],
        "allocations": {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    collection_plans.insert_one(doc)
    return doc

def save_plan_allocations(plan_id: str, goals: List[str], allocations: Dict[str, int]):
    collection_plans.update_one({"_id": plan_id}, {"$set": {"goals": goals, "allocations": allocations, "updated_at": datetime.utcnow()}})

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
    dfw = df[mask_week & (df["pomodoro_type"] == "Work")]
    if dfw.empty:
        return []
    if 'goal_id' not in dfw.columns:
        dfw = dfw.copy()
        dfw['goal_id'] = None
    by_goal = dfw.groupby(dfw["goal_id"].astype('object').fillna("NONE")).size().sort_values(ascending=False)
    total = int(by_goal.sum())
    if total < 4:
        return []
    top2 = by_goal.head(2).sum()
    if top2 / total >= threshold_pct:
        dominating = list(by_goal.head(2).index)
        dominating = [g for g in dominating if g != "NONE"]
        others = by_goal[~by_goal.index.isin(dominating)]
        if len(others) == 0 or any(others < min_other):
            return dominating
    return []

# === SESSION SAVE ===
def save_pomodoro_session(user: str, is_break: bool, duration: int, goal_id: Optional[str], task: str, category_label: str):
    now = now_ist()
    work_stream_type = "" if is_break else ("Goal" if goal_id else "Custom")
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
        "work_stream_type": work_stream_type,
        "created_at": datetime.utcnow()
    }
    collection_logs.insert_one(doc)
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
        "custom_categories": ["Learning", "Projects", "Research", "Planning"],
        "planning_week_date": now_ist().date(),
        "review_week_date": now_ist().date(),
        "analytics_mode": "Week Review",
        "break_extend_total": 0,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# === ADMIN / UTIL ===
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

# === HEADER ===
def main_header():
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
            "📓 Journal",
        ]
        st.session_state.page = st.selectbox("📍 Navigate", pages,
                                             index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
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

# === WEEKLY PLANNER PAGE ===
def render_weekly_planner():
    st.header("📅 Weekly Goal Planner")

    user = st.session_state.user
    pick_date = st.date_input("📆 Plan for week of", value=st.session_state.planning_week_date)
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
        total = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we},
                                        weekdays=wd_count, weekend_days=we_count)
        st.metric(f"Weekly Capacity ({week_start} → {week_end})", f"{total}")
        if (wp != settings["weekday_poms"]) or (we != settings["weekend_poms"]):
            if st.button("💾 Save Capacity", use_container_width=True):
                users_collection.update_one({"username": user},
                                            {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
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
    show = goals_df[["_id","title","goal_type","priority_weight","status"]].rename(
        columns={"_id":"Goal ID","goal_type":"Type","priority_weight":"Weight"})
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🧮 Auto-Allocate Weekly Pomodoros")
    wd_count, we_count = week_day_counts(week_start)
    total_poms = compute_weekly_capacity(get_user_settings(user), weekdays=wd_count, weekend_days=we_count)
    weight_map = {row["_id"]: int(row["priority_weight"]) for _, row in goals_df.iterrows()}
    auto = proportional_allocation(total_poms, weight_map)
    st.caption("Adjust numbers to fine-tune allocations (sum preserved).")

    prev_start = week_start - timedelta(days=7)
    prev_plan = collection_plans.find_one({"_id": f"{user}|{prev_start.isoformat()}"})
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
            val = st.number_input(f"{row['title']}", min_value=0, max_value=total_poms,
                                  value=default_val, step=1, key=f"alloc_{row['_id']}")
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

    if st.button("📌 Save Weekly Plan", type="primary"):
        save_plan_allocations(plan["_id"], list(edited.keys()), edited)
        st.success("Weekly plan saved!")

# === TIMER WIDGET (auto-breaks) ===
def render_timer_widget() -> bool:
    if not st.session_state.start_time:
        return False

    base_duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
    duration = base_duration + (st.session_state.break_extend_total if st.session_state.is_break else 0)
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

        if st.session_state.is_break:
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button("⏭️ Skip Break", use_container_width=True):
                    save_pomodoro_session(
                        user=st.session_state.user,
                        is_break=True,
                        duration=int((duration - remaining) / 60) if duration > remaining else BREAK_MIN,
                        goal_id=None, task="", category_label=""
                    )
                    sound_alert()
                    st.success("Break skipped.")
                    st.session_state.start_time = None
                    st.session_state.is_break = False
                    st.session_state.task = ""
                    st.session_state.active_goal_id = None
                    st.session_state.active_goal_title = ""
                    st.session_state.break_extend_total = 0
                    st.rerun()
            with c2:
                if st.button("➕ +5 min", use_container_width=True, disabled=st.session_state.break_extend_total >= 600):
                    st.session_state.break_extend_total = min(600, st.session_state.break_extend_total + 300)
                    st.rerun()
            with c3:
                st.caption("Auto break running…")

        st.info("🧘 Take a breather!" if st.session_state.is_break else "💪 Stay focused!")
        time.sleep(1)
        st.rerun()
        return True

    # timer finished
    if st.session_state.is_break:
        save_pomodoro_session(user=st.session_state.user, is_break=True, duration=int((duration)/60),
                              goal_id=None, task="", category_label="")
        sound_alert(); st.balloons(); st.success("🎉 Break complete!")
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""
        st.session_state.break_extend_total = 0
        return True
    else:
        # Work finished → auto-start break
        save_pomodoro_session(
            user=st.session_state.user, is_break=False, duration=POMODORO_MIN,
            goal_id=st.session_state.active_goal_id, task=st.session_state.task,
            category_label=st.session_state.active_goal_title
        )
        sound_alert(); st.balloons(); st.success("🎉 Work session complete! ☕ Starting break automatically.")
        st.session_state.start_time = time.time()
        st.session_state.is_break = True
        st.session_state.break_extend_total = 0
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""
        st.rerun()
        return True

# === FOCUS TIMER PAGE ===
def render_focus_timer():
    st.header("🎯 Focus Timer")
    if render_timer_widget():
        return

    user = st.session_state.user
    plan = get_or_create_weekly_plan(user, now_ist().date())

    df_all = get_user_sessions(user)
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df_all)
    render_daily_target_planner(df_all, today_progress)
    st.divider()

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
            title = r['title']; gid = r['_id']
            disabled = gid in locked
            alloc = plan.get('allocations', {}).get(gid, 0)
            label = f"{title}  ·  plan:{alloc}"
            choices.append((label, gid, disabled))

        c1, c2 = st.columns([1,2])
        with c1:
            options_labels = [lab + ("  🔒" if dis else "") for (lab,_,dis) in choices] or ["(no goals)"]
            selected_idx = st.selectbox("Weekly Goal", options=range(len(options_labels)),
                                        format_func=lambda i: options_labels[i], disabled=len(choices)==0)
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
            if st.button("☕ Start Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.session_state.break_extend_total = 0
                st.rerun()
    else:
        cat_options = st.session_state.custom_categories + ["➕ Add New"]
        selected = st.selectbox("📂 Custom Category", cat_options)
        if selected == "➕ Add New":
            new_cat = st.text_input("New custom category", placeholder="e.g., Marketing")
            if new_cat and st.button("✅ Add Category"):
                if new_cat not in st.session_state.custom_categories:
                    st.session_state.custom_categories.append(new_cat)
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
            if st.button("☕ Start Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.session_state.active_goal_id = None
                st.session_state.active_goal_title = ""
                st.session_state.task = ""
                st.session_state.break_extend_total = 0
                st.rerun()

    df = get_user_sessions(user)
    if not df.empty:
        today = now_ist().date()
        today_data = df[df["date"].dt.date == today]
        work_today = today_data[today_data["pomodoro_type"]=="Work"]
        breaks_today = len(today_data[today_data["pomodoro_type"]=="Break"])
        st.divider(); st.subheader("📊 Today's Summary")
        col1,col2,col3,col4 = st.columns(4)
        with col1: st.metric("Work Sessions", len(work_today))
        with col2: st.metric("Focus Minutes", int(work_today['duration'].sum()))
        with col3:
            ratio = breaks_today / max(1,len(work_today))
            label = "⚖️ Well balanced" if 0.3<=ratio<=0.7 else ("🎯 More focus" if ratio>0.7 else "🧘 Take breaks")
            st.metric("Breaks", breaks_today, help=label)
        with col4:
            current_target = get_daily_target(user)
            st.metric("Target Progress", f"{(len(work_today)/max(1,int(current_target)))*100:.0f}%" if current_target else "—")

# === ANALYTICS & REVIEW (merged) ===
def entropy_balance(counts: List[int]) -> float:
    """Return 0..100 (0=one goal dominates, 100=perfectly balanced)."""
    counts = [c for c in counts if c > 0]
    if not counts:
        return 0.0
    n = len(counts)
    total = sum(counts)
    p = [c/total for c in counts]
    H = -sum(pi*math.log(pi + 1e-12) for pi in p)
    Hmax = math.log(n)
    return 100.0 * (H / Hmax) if Hmax > 0 else 0.0

def render_analytics_review():
    st.header("📊 Analytics & Review")

    # Toggle: Week Review vs Trends
    st.session_state.analytics_mode = st.radio("Mode", ["Week Review", "Trends"],
                                               index=0 if st.session_state.analytics_mode=="Week Review" else 1,
                                               horizontal=True)

    user = st.session_state.user
    df = get_user_data(user)
    if df.empty:
        st.info("📈 Analytics will appear after your first session")
        return

    if st.session_state.analytics_mode == "Week Review":
        # Week picker
        pick_date = st.date_input("📆 Review week of", value=st.session_state.review_week_date, key="review_week_picker")
        week_start, week_end = week_bounds_ist(pick_date)
        if pick_date != st.session_state.review_week_date:
            st.session_state.review_week_date = pick_date
            st.rerun()

        plan = get_or_create_weekly_plan(user, week_start)

        # Filter week sessions
        mask_week = (df['date'].dt.date>=week_start) & (df['date'].dt.date<=week_end) & (df['pomodoro_type']=="Work")
        dfw = df[mask_week]
        if dfw.empty:
            st.info("No work sessions in this week.")
            return

        # --- KPIs ---
        # Plan adherence
        planned = plan.get('allocations', {}) or {}
        total_planned = int(sum(planned.values())) if planned else 0
        actual_goal = int(dfw['goal_id'].notna().sum())
        plan_adherence = (actual_goal / total_planned * 100.0) if total_planned > 0 else 0.0

        # Custom share
        custom_work = int((dfw['work_stream_type']=="Custom").sum())
        total_work = int(len(dfw))
        custom_share = (custom_work / max(1,total_work) * 100.0)

        # Deep work %
        deep_work = int((dfw['duration'] >= 23).sum())
        deep_pct = deep_work / max(1, total_work) * 100.0

        # Balance score (across goals only)
        by_goal = dfw[dfw['goal_id'].notna()].groupby('goal_id').size()
        bal_score = entropy_balance(list(by_goal.values)) if len(by_goal) > 0 else 0.0

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Plan Adherence", f"{plan_adherence:.0f}%")
        with c2: st.metric("Custom Share", f"{custom_share:.0f}%")
        with c3: st.metric("Deep-work %", f"{deep_pct:.0f}%")
        with c4: st.metric("Balance Score", f"{bal_score:.0f}")

        st.divider()

        # Goals vs Custom donut
        donut_df = pd.DataFrame({
            "stream": ["Goals","Custom"],
            "sessions": [total_work - custom_work, custom_work]
        })
        fig_mix = px.pie(donut_df, names="stream", values="sessions", hole=0.45,
                         title=f"Goals vs Custom — {week_start} → {week_end}",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_mix.update_layout(height=320, title_x=0.5)
        st.plotly_chart(fig_mix, use_container_width=True)

        st.subheader("Plan vs Actual (by Goal)")
        def title_of(gid):
            if gid=='NONE' or gid is None:
                return 'Unassigned'
            doc = collection_goals.find_one({"_id": gid})
            return doc['title'] if doc else '(missing)'

        actual_by_goal = dfw.groupby(dfw['goal_id'].fillna('NONE')).size().rename('actual').reset_index()
        actual_by_goal['title'] = actual_by_goal['goal_id'].apply(title_of)

        planned_df = pd.DataFrame([{"goal_id": gid, "planned": v, "title": title_of(gid)} for gid, v in planned.items()])
        m = pd.merge(planned_df, actual_by_goal, on=['goal_id','title'], how='outer').fillna(0)
        m['planned'] = m['planned'].astype(int)
        m['actual'] = m['actual'].astype(int)

        cL, cR = st.columns([2,1])
        with cL:
            if not m.empty:
                fig = px.bar(m, x='title', y=['planned','actual'], barmode='group',
                             title='Planned vs Actual Pomodoros')
                fig.update_layout(height=380, xaxis_title='', legend_title='')
                st.plotly_chart(fig, use_container_width=True)
        with cR:
            st.metric("Planned (week)", int(sum(planned.values())) if planned else 0)
            st.metric("Actual (week)", int(m['actual'].sum()))
            st.metric("Δ Actual - Planned", int(m['actual'].sum()) - (int(sum(planned.values())) if planned else 0))

        # Insights (simple, actionable)
        st.markdown("### 💡 Insights")
        if total_planned > 0 and plan_adherence < 85:
            st.warning("Plan adherence is low. Consider scheduling catch-up blocks or lowering next week’s allocations.")
        if custom_share > 30:
            st.info("Custom work is high. Add a ‘Custom buffer’ next week or formalize frequent custom categories as goals.")
        if bal_score < 50 and total_work >= 6:
            st.warning("One or two goals dominate. Use the lock window or shift a couple of sessions to under-served goals.")
        if deep_pct < 40:
            st.info("Deep-work is low. Try batching interruptions and protecting morning focus blocks.")

        st.divider()
        st.subheader("Close Out & Rollover")
        if not m.empty:
            for _, row in m.iterrows():
                gid = row['goal_id']
                if gid=='NONE' or gid is None:
                    continue
                col1, col2, col3, col4 = st.columns([3,2,2,2])
                with col1: st.write(f"**{row['title']}**")
                with col2:
                    status = st.selectbox("Status",
                        ["Completed","Rollover","On Hold","Archived","In Progress"],
                        index=4, key=f"close_{gid}_{week_start}")
                with col3:
                    carry = max(0, int(row['planned']) - int(row['actual']))
                    carry = st.number_input("Carry fwd poms", 0, 200, value=carry, key=f"carry_{gid}_{week_start}")
                with col4:
                    if st.button("✅ Apply", key=f"apply_{gid}_{week_start}"):
                        collection_goals.update_one(
                            {"_id": gid},
                            {"$set": {"status": "Completed" if status=="Completed"
                                      else ("On Hold" if status=="On Hold"
                                      else ("Archived" if status=="Archived" else "In Progress"))}}
                        )
                        if status=="Rollover" and carry>0:
                            next_start = week_start + timedelta(days=7)
                            next_plan = get_or_create_weekly_plan(user, next_start)
                            next_alloc = next_plan.get('allocations', {})
                            next_goals = set(next_plan.get('goals', []))
                            next_goals.add(gid)
                            next_alloc[gid] = next_alloc.get(gid, 0) + int(carry)
                            save_plan_allocations(next_plan['_id'], list(next_goals), next_alloc)
                        st.success("Updated")

    else:
        # === Trends (uses existing analytics content) ===
        df_work = df[df["pomodoro_type"] == "Work"]
        today = now_ist().date()

        st.subheader("📈 Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🎯 Total Sessions", len(df_work))
        with col2: st.metric("⏱️ Total Hours", int(df_work['duration'].sum() // 60))
        with col3:
            active_days = len(df_work.groupby(df_work["date"].dt.date).size())
            st.metric("📅 Active Days", int(active_days))
        with col4:
            if len(df_work) > 0:
                avg_daily = df_work.groupby(df_work["date"].dt.date).size().mean()
                st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

        st.divider()
        st.subheader("📈 Daily Performance (Last 30 Days)")
        daily_data = []
        for i in range(30):
            date_check = today - timedelta(days=29-i)
            daily_work = df_work[df_work["date"].dt.date == date_check]
            daily_data.append({'date': date_check.strftime('%m/%d'),
                               'minutes': int(daily_work['duration'].sum())})
        daily_df = pd.DataFrame(daily_data)
        if daily_df['minutes'].sum() > 0:
            fig = px.bar(daily_df, x='date', y='minutes', title="Daily Focus Minutes",
                         color='minutes', color_continuous_scale='Blues')
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.divider(); st.subheader("🎯 Time Investment Analysis")
        time_filter = st.selectbox("📅 Time Period", ["Last 7 days", "Last 30 days", "All time"], index=1)
        if time_filter == "Last 7 days":
            cutoff_date = today - timedelta(days=7)
            filtered_work = df_work[df_work["date"].dt.date >= cutoff_date]
        elif time_filter == "Last 30 days":
            cutoff_date = today - timedelta(days=30)
            filtered_work = df_work[df_work["date"].dt.date >= cutoff_date]
        else:
            filtered_work = df_work
        if filtered_work.empty:
            st.info(f"📊 No data available for {time_filter.lower()}")
            return

        st.markdown("### 📂 Category Deep Dive")
        category_stats = filtered_work.groupby('category').agg({'duration': ['sum', 'count', 'mean']}).round(1)
        category_stats.columns = ['total_minutes', 'sessions', 'avg_session']
        category_stats = category_stats.sort_values('total_minutes', ascending=False)

        col1, col2 = st.columns([3,2])
        with col1:
            if len(category_stats) > 0:
                total_time = category_stats['total_minutes'].sum()
                fig_donut = px.pie(values=category_stats['total_minutes'], names=category_stats.index,
                                   title=f"📊 Time Distribution by Category ({time_filter})",
                                   hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
                total_hours = int(total_time) // 60
                total_mins = int(total_time) % 60
                center_text = f"{total_hours}h {total_mins}m" if total_hours > 0 else f"{total_mins}m"
                fig_donut.add_annotation(text=f"<b>Total</b><br>{center_text}", x=0.5, y=0.5, showarrow=False)
                fig_donut.update_layout(height=380, showlegend=True, title_x=0.5)
                st.plotly_chart(fig_donut, use_container_width=True)
        with col2:
            st.markdown("#### 📈 Category Performance")
            performance_data = []
            for cat in category_stats.index:
                total_mins = category_stats.loc[cat, 'total_minutes']
                sessions = int(category_stats.loc[cat, 'sessions'])
                avg_session = category_stats.loc[cat, 'avg_session']
                if total_mins >= 60:
                    hours = int(total_mins // 60); mins = int(total_mins % 60)
                    time_str = f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
                else:
                    time_str = f"{int(total_mins)}m"
                performance_data.append({'Category': cat, 'Time': time_str, 'Sessions': sessions,
                                         'Avg/Session': f"{avg_session:.0f}m"})
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True,
                         height=min(len(perf_df)*35+38, 300))

        st.markdown("### 🎯 Task Performance Analysis")
        task_stats = filtered_work.groupby(['category','task']).agg({'duration':['sum','count','mean']}).round(1)
        task_stats.columns = ['total_minutes','sessions','avg_session']
        task_stats = task_stats.reset_index().sort_values('total_minutes', ascending=False)
        col1, col2 = st.columns([3,2])
        with col1:
            top_tasks = task_stats.head(12)
            if len(top_tasks) > 0:
                fig_tasks = px.bar(top_tasks, x='total_minutes', y='task', color='category',
                                   title=f"🎯 Top Tasks by Time Investment ({time_filter})",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                fig_tasks.update_layout(height=max(380, len(top_tasks)*30),
                                        yaxis={'categoryorder':'total ascending'},
                                        title_x=0.5, showlegend=True)
                st.plotly_chart(fig_tasks, use_container_width=True)
        with col2:
            st.markdown("#### 💡 Smart Insights & Recommendations")
            if len(task_stats) > 0:
                total_time_invested = task_stats['total_minutes'].sum()
                top_task = task_stats.iloc[0]
                top_task_pct = (top_task['total_minutes']/max(1,total_time_invested))*100
                if top_task_pct > 50:
                    st.warning("⚖️ One task is dominating your time. Consider rebalancing.")
                elif top_task_pct > 25:
                    st.info("🎯 Clear primary task focus this period.")
                else:
                    st.success("✅ Time is well distributed across tasks.")

        st.divider(); st.subheader("🔥 Consistency Tracking")
        daily_counts = df_work.groupby(df_work["date"].dt.date).size()
        active_days = len(daily_counts)
        min_sessions = 1 if active_days <= 12 else 2
        today_d = today
        current_streak = 0
        for i in range(365):
            check_date = today_d - timedelta(days=i)
            if daily_counts.get(check_date, 0) >= min_sessions:
                current_streak += 1
            else:
                break
        col1,col2,col3 = st.columns(3)
        with col1: st.metric("🔥 Current Streak", f"{current_streak} days")
        with col2:
            max_streak = 0; tmp = 0
            for i in range(365):
                check_date = today_d - timedelta(days=i)
                if daily_counts.get(check_date, 0) >= min_sessions:
                    tmp += 1; max_streak = max(max_streak, tmp)
                else:
                    tmp = 0
            st.metric("🏆 Best Streak", f"{max_streak} days")
        with col3:
            recent_days = [daily_counts.get(today_d - timedelta(days=i), 0) for i in range(7)]
            consistency = len([d for d in recent_days if d >= min_sessions]) / 7 * 100
            st.metric("📊 Weekly Consistency", f"{consistency:.0f}%")

# === JOURNAL (Reflection + Notes merged) ===
def add_note(content: str, d: str, user: str):
    nid = hashlib.sha256(f"{d}_{content}_{user}".encode()).hexdigest()
    doc = {"_id": nid, "type":"Note", "date": d, "content": content, "user": user, "created_at": datetime.utcnow()}
    collection_logs.update_one({"_id": nid}, {"$set": doc}, upsert=True)

def render_journal():
    st.header("📓 Journal")

    user = st.session_state.user
    today_iso = now_ist().date().isoformat()

    colA, colB = st.columns([1,1])
    with colA:
        st.subheader("🧠 End-of-Day Reflection")
        ref_date = st.date_input("Date", now_ist().date(), key="journal_ref_date")
        with st.form("reflection_form", clear_on_submit=True):
            aligned = st.selectbox("Aligned with plan?", ["Yes","Partly","No"])
            rating = st.slider("Focus quality (1-5)", 1, 5, 3)
            blockers = st.text_area("Blockers / distractions")
            notes = st.text_area("Insights / anything to note")
            submitted = st.form_submit_button("💾 Save Reflection")
            if submitted:
                collection_reflections.update_one(
                    {"user": user, "date": ref_date.isoformat()},
                    {"$set": {
                        "user": user,
                        "date": ref_date.isoformat(),
                        "aligned": aligned,
                        "focus_rating": int(rating),
                        "blockers": blockers.strip(),
                        "notes": notes.strip(),
                        "created_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                st.success("Reflection saved ✨")

        recent_refs = list(collection_reflections.find({"user": user}).sort("date", -1).limit(7))
        if recent_refs:
            st.markdown("#### Recent Reflections")
            rdf = pd.DataFrame(recent_refs)
            st.dataframe(rdf[["date","aligned","focus_rating","blockers","notes"]],
                         use_container_width=True, hide_index=True, height=240)

    with colB:
        st.subheader("📝 Quick Note")
        note_date = st.date_input("Note date", now_ist().date(), key="journal_note_date")
        note_text = st.text_area("Your note...", height=140, key="journal_note_text")
        c1, c2 = st.columns([1,2])
        with c1:
            if st.button("💾 Save Note"):
                if note_text.strip():
                    add_note(note_text.strip(), note_date.isoformat(), user)
                    st.success("Note saved")
                else:
                    st.warning("Add some content")
        with c2:
            st.caption("Tip: capture blockers or ideas you don’t want to lose.")

    st.divider()
    st.subheader("🔎 Browse Notes & Reflections")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("From", now_ist().date()-timedelta(days=7), key="journal_from")
    with c2:
        end = st.date_input("To", now_ist().date(), key="journal_to")

    # Notes
    notes = list(collection_logs.find(
        {"type":"Note","user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
    ).sort("date", -1))
    refs = list(collection_reflections.find(
        {"user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
    ).sort("date", -1))

    cN, cR = st.columns(2)
    with cN:
        st.markdown("#### Notes")
        if notes:
            for n in notes:
                st.write(f"**{n['date']}** — {n['content']}")
                st.divider()
        else:
            st.info("No notes in this range")
    with cR:
        st.markdown("#### Reflections")
        if refs:
            for r in refs:
                st.write(f"**{r['date']}** — aligned: {r.get('aligned','?')} • rating: {r.get('focus_rating','-')}")
                bl = r.get('blockers',''); nt = r.get('notes','')
                if bl: st.caption(f"Blockers: {bl}")
                if nt: st.caption(f"Notes: {nt}")
                st.divider()
        else:
            st.info("No reflections in this range")

# === MAIN ===
def main():
    main_header()
    st.divider()
    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer()
    elif page == "📅 Weekly Planner":
        render_weekly_planner()
    elif page == "📊 Analytics & Review":
        render_analytics_review()
    elif page == "📓 Journal":
        render_journal()

if __name__ == "__main__":
    main()
