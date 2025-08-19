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
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/blob/main/one_piece_overtake.mp3"

# === DB INIT ===

def sound_alert():
    """Play a completion sound (browser will auto-play if allowed)."""
    st.components.v1.html(f"""
        <audio autoplay><source src=\"{SOUND_PATH}\" type=\"audio/mpeg\"></audio>
        <script>
            const audio = new Audio('{SOUND_PATH}');
            audio.volume = 0.6;
            audio.play().catch(()=>{});
        </script>
    """, height=0)

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
collection_logs = db["logs"]                 # existing sessions/targets/notes
users_collection = db["users"]               # user settings
collection_goals = db["goals"]               # new: weekly goals catalog
collection_plans = db["weekly_plans"]        # new: plan per week
collection_reflections = db["reflections"]   # new: daily reflections

# === HELPERS: TIME / WEEK ===
def now_ist() -> datetime:
    return datetime.now(IST)

def week_bounds_ist(d: date) -> Tuple[date, date]:
    # Monday start (ISO), Sunday end
    weekday = d.weekday()  # 0=Mon
    start = d - timedelta(days=weekday)
    end = start + timedelta(days=6)
    return start, end

def current_week_id(d: Optional[date] = None) -> str:
    if d is None:
        d = now_ist().date()
    start, _ = week_bounds_ist(d)
    return f"{start.isoformat()}"

# === CACHING LAYERS ===
@st.cache_data(ttl=300)
def get_user_sessions(username: str) -> pd.DataFrame:
    recs = list(collection_logs.find({"type": "Pomodoro", "user": username}))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    return df

# compat wrapper for analytics lifted from original app
@st.cache_data(ttl=300)
def get_user_data(username: str) -> pd.DataFrame:
    return get_user_sessions(username)

@st.cache_data(ttl=120)
def get_user_settings(username: str) -> Dict:
    doc = users_collection.find_one({"username": username})
    if not doc:
        # seed defaults
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

# === CORE: USERS ===
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
    # Generate deterministic goal id per user+title
    gid = hashlib.sha256(f"{username}|{title}".encode()).hexdigest()[:16]
    # Only put non-overlapping fields in $setOnInsert to avoid operator path conflicts.
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
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=["_id","user","title","priority_weight","goal_type","status","target_poms","poms_completed"]) 

# === WEEKLY PLAN ===
def compute_weekly_capacity(settings: Dict, weekdays: int = 5, weekend_days: int = 2) -> int:
    return settings["weekday_poms"] * weekdays + settings["weekend_poms"] * weekend_days

def proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    total_w = max(1, sum(max(1, int(w)) for w in weights.values()))
    raw = {gid: (max(1, int(w)) / total_w) * total for gid, w in weights.items()}
    # round while preserving sum
    allocated = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(allocated.values())
    # distribute remainder by largest fractional part
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
    # create skeleton plan with zero goals until user sets it
    settings = get_user_settings(username)
    total_poms = compute_weekly_capacity(settings)
    doc = {
        "_id": pid,
        "user": username,
        "week_start": start.isoformat(),
        "week_end": end.isoformat(),
        "total_poms": total_poms,
        "goals": [],                 # list of goal_ids
        "allocations": {},           # goal_id -> planned pomodoros
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
    # If within lock window, check distribution
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
    # count sessions per goal_id
    by_goal = dfw.groupby(dfw["goal_id"].fillna("NONE")).size().sort_values(ascending=False)
    total = int(by_goal.sum())
    if total < 4:
        return []
    top2 = by_goal.head(2).sum()
    if top2 / total >= threshold_pct:
        # find goals that are dominating (top1/top2) and lock them
        dominating = list(by_goal.head(2).index)
        # only lock real goals (skip NONE)
        dominating = [g for g in dominating if g != "NONE"]
        # check if other goals reached min_other
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
        # keep compatibility with old analytics: use goal title as category when present
        "category": category_label if (category_label and not is_break) else "",
        "created_at": datetime.utcnow()
    }
    collection_logs.insert_one(doc)
    # invalidate caches
    get_user_sessions.clear()

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
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# === HEADER ===
def render_header():
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
            "🧠 Reflection",
            "🧭 Weekly Review",
            "📊 Analytics",
            "📝 Notes Saver",
            "🗂️ Notes Viewer",
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

# === WEEKLY PLANNER PAGE ===
def render_weekly_planner():
    st.header("📅 Weekly Goal Planner")

    user = st.session_state.user
    settings = get_user_settings(user)
    week_id = current_week_id()
    plan = get_or_create_weekly_plan(user)

    colA, colB, colC = st.columns(3)
    with colA:
        wp = st.number_input("Avg Pomodoros / Weekday", 0, 12, value=settings["weekday_poms"])
    with colB:
        we = st.number_input("Avg Pomodoros / Weekend Day", 0, 12, value=settings["weekend_poms"])
    with colC:
        total = compute_weekly_capacity({"weekday_poms": wp, "weekend_poms": we})
        st.metric("Estimated Weekly Capacity", f"{total}")
        if (wp != settings["weekday_poms"]) or (we != settings["weekend_poms"]):
            if st.button("💾 Save Capacity", use_container_width=True):
                users_collection.update_one({"username": user}, {"$set": {"weekday_poms": int(wp), "weekend_poms": int(we)}})
                get_user_settings.clear()
                plan = get_or_create_weekly_plan(user)  # refresh totals next rerun
                st.success("Updated!")
                st.rerun()

    st.divider()

    st.subheader("🎯 Define Priorities (Top 3–4)")
    # Add / edit goals
    with st.expander("➕ Add or Update Goal", expanded=False):
        g_title = st.text_input("Title", placeholder="e.g., UGC NET Paper 1")
        g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
        g_weight = st.select_slider("Priority Weight", options=[1,2,3], value=2, help="High=3, Medium=2, Low=1")
        g_status = st.selectbox("Status", ["New","In Progress","Completed","On Hold","Archived"], index=0)
        if st.button("💾 Save Goal"):
            if g_title.strip():
                gid = upsert_goal(user, g_title.strip(), int(g_weight), g_type, g_status)
                fetch_goals.clear()
                st.success("Saved goal")
                st.rerun()
            else:
                st.warning("Please provide a title")

    goals_df = fetch_goals(user, statuses=["New","In Progress"])  # active
    if goals_df.empty:
        st.info("Add 3–4 goals above to plan this week.")
        return

    st.markdown("#### Active Goals")
    show = goals_df[["_id","title","goal_type","priority_weight","status"]].rename(columns={"_id":"Goal ID","goal_type":"Type","priority_weight":"Weight"})
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🧮 Auto-Allocate Weekly Pomodoros")
    total_poms = compute_weekly_capacity(get_user_settings(user))
    weight_map = {row["_id"]: int(row["priority_weight"]) for _, row in goals_df.iterrows()}
    auto = proportional_allocation(total_poms, weight_map)

    st.caption("Adjust sliders to fine-tune allocations (sum preserved).")

    # Sliders
    edited = {}
    cols = st.columns(min(4, len(auto)))
    i = 0
    for _, row in goals_df.iterrows():
        with cols[i % len(cols)]:
            val = st.number_input(f"{row['title']}", min_value=0, max_value=total_poms, value=int(auto[row['_id']]), step=1, key=f"alloc_{row['_id']}")
            edited[row["_id"]] = int(val)
        i += 1

    # Normalize to sum total_poms
    sum_edit = sum(edited.values())
    if sum_edit != total_poms:
        st.warning(f"Allocations sum to {sum_edit}, not {total_poms}. Click to auto-correct.")
        if st.button("🔁 Normalize to Total"):
            edited = proportional_allocation(total_poms, {gid: max(1, v) for gid, v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"alloc_{gid}"] = v
            st.experimental_rerun()

    if st.button("📌 Save Weekly Plan", type="primary"):
        save_plan_allocations(plan["_id"], list(edited.keys()), edited)
        st.success("Weekly plan saved!")

# === FOCUS TIMER PAGE (with Goal selection + Locking) ===
def render_timer_widget() -> bool:
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
        # complete
        label = st.session_state.active_goal_title
        save_pomodoro_session(
            user=st.session_state.user,
            is_break=st.session_state.is_break,
            duration=BREAK_MIN if st.session_state.is_break else POMODORO_MIN,
            goal_id=st.session_state.active_goal_id,
            task=st.session_state.task,
            category_label=label
        )
        sound_alert()
        st.balloons(); st.success("🎉 Session complete!")
        st.session_state.start_time = None
        st.session_state.is_break = False
        st.session_state.task = ""
        st.session_state.active_goal_id = None
        st.session_state.active_goal_title = ""
        return True


def render_focus_timer():
    st.header("🎯 Focus Timer")
    if render_timer_widget():
        return

    user = st.session_state.user
    plan = get_or_create_weekly_plan(user)

    # Daily Target Planner
    df_all = get_user_sessions(user)
    today_progress, adaptive_goal, today_minutes = render_daily_goal(df_all)
    render_daily_target_planner(df_all, today_progress)
    st.divider()

    # Choose mode: Weekly Goal vs Custom Category (unassigned)
    mode = st.radio("Mode", ["Weekly Goal", "Custom Category"], horizontal=True)

    if mode == "Weekly Goal":
        # Goal selection respecting locks
        active_goal_ids = plan.get("goals", [])
        if not active_goal_ids:
            st.warning("No weekly plan saved yet. Please create allocations in the **Weekly Planner** page.")
        goals_df = fetch_goals(user, statuses=["New","In Progress"]) 
        goals_df = goals_df[goals_df["_id"].isin(active_goal_ids)] if not goals_df.empty else goals_df

        locked = set(locked_goals_for_user_plan(user, plan))
        if locked:
            st.warning("⚖️ Balanced Focus: Top goals temporarily locked. Spend a minimum on other goals to unlock.")

        # Build choices
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
        # Custom category flow (unassigned goal_id)
        cat_options = st.session_state.custom_categories + ["+ Add New"]
        selected = st.selectbox("📂 Category", cat_options)
        if selected == "+ Add New":
            new_cat = st.text_input("New category", placeholder="e.g., Marketing")
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
        today_data = df[df["date"].dt.date == today]
        work_today = today_data[today_data["pomodoro_type"]=="Work"]
        breaks_today = len(today_data[today_data["pomodoro_type"]=="Break"])
        st.divider(); st.subheader("📊 Today's Summary")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("Work Sessions", len(work_today))
        with col2:
            st.metric("Focus Minutes", int(work_today['duration'].sum()))
        with col3:
            ratio = breaks_today / max(1,len(work_today))
            label = "⚖️ Well balanced" if 0.3<=ratio<=0.7 else ("🎯 More focus" if ratio>0.7 else "🧘 Take breaks")
            st.metric("Breaks", breaks_today, help=label)
        with col4:
            current_target = get_daily_target(user)
            if current_target:
                pct = (len(work_today)/max(1,int(current_target)))*100
                st.metric("Target Progress", f"{pct:.0f}%")
            else:
                st.metric("Target Progress", "—")

# === REFLECTION PAGE ===

def render_reflection_page():
    st.header("🧠 End-of-Day Reflection")
    user = st.session_state.user
    today = now_ist().date().isoformat()

    with st.form("reflection_form", clear_on_submit=True):
        aligned = st.selectbox("Was today aligned with the weekly plan?", ["Yes","Partly","No"])
        rating = st.slider("Focus quality (1-5)", 1, 5, 3)
        blockers = st.text_area("Blockers / distractions")
        notes = st.text_area("Insights / anything to note")
        submitted = st.form_submit_button("💾 Save Reflection")
        if submitted:
            collection_reflections.update_one(
                {"user": user, "date": today},
                {"$set": {
                    "user": user,
                    "date": today,
                    "aligned": aligned,
                    "focus_rating": int(rating),
                    "blockers": blockers.strip(),
                    "notes": notes.strip(),
                    "created_at": datetime.utcnow()
                }},
                upsert=True
            )
            st.success("Saved ✨")

    # Recent reflections
    recs = list(collection_reflections.find({"user": user}).sort("date", -1).limit(7))
    if recs:
        st.subheader("Recent")
        df = pd.DataFrame(recs)
        st.dataframe(df[["date","aligned","focus_rating","blockers","notes"]], use_container_width=True, hide_index=True)

# === WEEKLY REVIEW PAGE ===

def render_weekly_review():
    st.header("🧭 Weekly Review & Transition")
    user = st.session_state.user
    plan = get_or_create_weekly_plan(user)

    # Planned vs Actual
    st.subheader("Plan vs Actual")
    df = get_user_sessions(user)
    if df.empty:
        st.info("No sessions yet.")
        return
    start = datetime.fromisoformat(plan['week_start']).date()
    end = datetime.fromisoformat(plan['week_end']).date()
    mask_week = (df['date'].dt.date>=start) & (df['date'].dt.date<=end) & (df['pomodoro_type']=="Work")
    dfw = df[mask_week]

    if dfw.empty:
        st.info("No work sessions this week yet.")
        return

    by_goal = dfw.groupby(dfw['goal_id'].fillna('NONE')).size().rename('actual').reset_index()
    # attach titles
    def title_of(gid):
        if gid=='NONE' or gid is None:
            return 'Unassigned'
        doc = collection_goals.find_one({"_id": gid})
        return doc['title'] if doc else '(missing)'
    by_goal['title'] = by_goal['goal_id'].apply(title_of)

    planned = plan.get('allocations', {})
    planned_df = pd.DataFrame([{"goal_id": gid, "planned": v, "title": title_of(gid)} for gid, v in planned.items()])

    m = pd.merge(planned_df, by_goal, on=['goal_id','title'], how='outer').fillna(0)
    m['planned'] = m['planned'].astype(int)
    m['actual'] = m['actual'].astype(int)

    c1, c2 = st.columns(2)
    with c1:
        if not m.empty:
            fig = px.bar(m, x='title', y=['planned','actual'], barmode='group', title='Planned vs Actual Pomodoros')
            fig.update_layout(height=360, xaxis_title='', legend_title='')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        total_planned = int(sum(planned.values())) if planned else 0
        total_actual = int(m['actual'].sum())
        st.metric("Planned (week)", total_planned)
        st.metric("Actual (week)", total_actual)
        delta = total_actual - total_planned
        st.metric("Δ Actual - Planned", delta)

    st.divider()

    # Close out goals
    st.subheader("Close Out & Rollover")
    if not m.empty:
        for _, row in m.iterrows():
            gid = row['goal_id']
            if gid=='NONE' or gid is None:
                continue
            col1, col2, col3, col4 = st.columns([3,2,2,2])
            with col1:
                st.write(f"**{row['title']}**")
            with col2:
                status = st.selectbox("Status", ["Completed","Rollover","On Hold","Archived","In Progress"], index=4, key=f"close_{gid}")
            with col3:
                carry = max(0, int(row['planned']) - int(row['actual']))
                carry = st.number_input("Carry fwd poms", 0, 200, value=carry, key=f"carry_{gid}")
            with col4:
                if st.button("✅ Apply", key=f"apply_{gid}"):
                    # update goal status
                    collection_goals.update_one({"_id": gid}, {"$set": {"status": "Completed" if status=="Completed" else ("On Hold" if status=="On Hold" else ("Archived" if status=="Archived" else "In Progress"))}})
                    # prepare next week plan if rollover
                    if status=="Rollover" and carry>0:
                        # next week plan skeleton
                        next_start = datetime.fromisoformat(plan['week_start']).date() + timedelta(days=7)
                        next_plan = get_or_create_weekly_plan(user, next_start)
                        next_alloc = next_plan.get('allocations', {})
                        next_goals = set(next_plan.get('goals', []))
                        next_goals.add(gid)
                        next_alloc[gid] = next_alloc.get(gid, 0) + int(carry)
                        save_plan_allocations(next_plan['_id'], list(next_goals), next_alloc)
                    st.success("Updated")

# === ANALYTICS (reuse but add goal views) ===

def render_analytics_page():
    """Rich analytics from your original app, compatible with goals via 'category' label."""
    st.header("📊 Analytics Dashboard")
    df = get_user_data(st.session_state.user)
    if df.empty:
        st.info("📈 Analytics will appear after your first session")
        return
    df_work = df[df["pomodoro_type"] == "Work"]
    today = now_ist().date()

    # Key metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Total Sessions", len(df_work))
    with col2:
        total_hours = df_work['duration'].sum() // 60
        st.metric("⏱️ Total Hours", int(total_hours))
    with col3:
        active_days = len(df_work.groupby(df_work["date"].dt.date).size())
        st.metric("📅 Active Days", int(active_days))
    with col4:
        if len(df_work) > 0:
            avg_daily = df_work.groupby(df_work["date"].dt.date).size().mean()
            st.metric("📊 Avg Daily", f"{avg_daily:.1f}")

    st.divider()

    # Daily performance chart (Last 30d)
    st.subheader("📈 Daily Performance (Last 30 Days)")
    daily_data = []
    for i in range(30):
        date_check = today - timedelta(days=29-i)
        daily_work = df_work[df_work["date"].dt.date == date_check]
        daily_data.append({'date': date_check.strftime('%m/%d'), 'sessions': len(daily_work), 'minutes': int(daily_work['duration'].sum())})
    daily_df = pd.DataFrame(daily_data)
    if daily_df['minutes'].sum() > 0:
        fig = px.bar(daily_df, x='date', y='minutes', title="Daily Focus Minutes", color='minutes', color_continuous_scale='Blues')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Time Investment Analysis
    st.divider(); st.subheader("🎯 Time Investment Analysis")
    col_filter1, _, _ = st.columns(3)
    with col_filter1:
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

    # Category Deep Dive (we use 'category' which is the goal title when present)
    st.markdown("### 📂 Category Deep Dive")
    category_stats = filtered_work.groupby('category').agg({'duration': ['sum', 'count', 'mean']}).round(1)
    category_stats.columns = ['total_minutes', 'sessions', 'avg_session']
    category_stats = category_stats.sort_values('total_minutes', ascending=False)

    col1, col2 = st.columns([3,2])
    with col1:
        if len(category_stats) > 0:
            total_time = category_stats['total_minutes'].sum()
            fig_donut = px.pie(values=category_stats['total_minutes'], names=category_stats.index, title=f"📊 Time Distribution by Category ({time_filter})", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            total_hours = int(total_time) // 60
            total_mins = int(total_time) % 60
            center_text = f"{total_hours}h {total_mins}m" if total_hours > 0 else f"{total_mins}m"
            fig_donut.add_annotation(text=f"<b>Total</b><br>{center_text}", x=0.5, y=0.5, showarrow=False)
            fig_donut.update_layout(height=400, showlegend=True, title_x=0.5)
            st.plotly_chart(fig_donut, use_container_width=True)
    with col2:
        st.markdown("#### 📈 Category Performance")
        performance_data = []
        for cat in category_stats.index:
            total_mins = category_stats.loc[cat, 'total_minutes']
            sessions = int(category_stats.loc[cat, 'sessions'])
            avg_session = category_stats.loc[cat, 'avg_session']
            # pretty time
            if total_mins >= 60:
                hours = int(total_mins // 60); mins = int(total_mins % 60)
                time_str = f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
            else:
                time_str = f"{int(total_mins)}m"
            performance_data.append({'Category': cat, 'Time': time_str, 'Sessions': sessions, 'Avg/Session': f"{avg_session:.0f}m"})
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True, height=min(len(perf_df)*35+38, 300))

    # Task Performance Analysis
    st.markdown("### 🎯 Task Performance Analysis")
    task_stats = filtered_work.groupby(['category','task']).agg({'duration':['sum','count','mean']}).round(1)
    task_stats.columns = ['total_minutes','sessions','avg_session']
    task_stats = task_stats.reset_index().sort_values('total_minutes', ascending=False)
    col1, col2 = st.columns([3,2])
    with col1:
        top_tasks = task_stats.head(12)
        if len(top_tasks) > 0:
            fig_tasks = px.bar(top_tasks, x='total_minutes', y='task', color='category', title=f"🎯 Top Tasks by Time Investment ({time_filter})", color_discrete_sequence=px.colors.qualitative.Set3)
            fig_tasks.update_layout(height=max(400, len(top_tasks)*30), yaxis={'categoryorder':'total ascending'}, title_x=0.5, showlegend=True)
            st.plotly_chart(fig_tasks, use_container_width=True)
    with col2:
        st.markdown("#### 💡 Smart Insights & Recommendations")
        if len(task_stats) > 0:
            total_time_invested = task_stats['total_minutes'].sum()
            top_task = task_stats.iloc[0]
            top_task_pct = (top_task['total_minutes']/max(1,total_time_invested))*100
            # simple insights
            if top_task_pct > 50:
                st.warning("⚖️ One task is dominating your time. Consider rebalancing.")
            elif top_task_pct > 25:
                st.info("🎯 Clear primary task focus this period.")
            else:
                st.success("✅ Time is well distributed across tasks.")

    # Weekly trends (stacked by category)
    if time_filter != "Last 7 days" and len(filtered_work) > 7:
        st.markdown("### 📊 Weekly Category Trends")
        filtered_work['week'] = filtered_work['date'].dt.isocalendar().week
        filtered_work['year_week'] = filtered_work['date'].dt.strftime('%Y-W%U')
        weekly_categories = filtered_work.groupby(['year_week','category'])['duration'].sum().reset_index()
        if len(weekly_categories) > 0:
            fig_weekly = px.bar(weekly_categories, x='year_week', y='duration', color='category', title="📈 Weekly Time Distribution by Category", color_discrete_sequence=px.colors.qualitative.Set3)
            fig_weekly.update_layout(height=350, xaxis_title="Week", yaxis_title="Time (minutes)", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_weekly, use_container_width=True)

    # Streak / consistency
    st.divider(); st.subheader("🔥 Consistency Tracking")
    daily_counts = df_work.groupby(df_work["date"].dt.date).size()
    active_days = len(daily_counts)
    min_sessions = 1 if active_days <= 12 else 2
    today_counts = daily_counts.to_dict()
    today_d = today
    current_streak = 0
    for i in range(365):
        check_date = today_d - timedelta(days=i)
        if today_counts.get(check_date, 0) >= min_sessions:
            current_streak += 1
        else:
            break
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("🔥 Current Streak", f"{current_streak} days")
    with col2:
        max_streak = 0; temp = 0
        for i in range(365):
            check_date = today_d - timedelta(days=i)
            if today_counts.get(check_date, 0) >= min_sessions:
                temp += 1; max_streak = max(max_streak, temp)
            else:
                temp = 0
        st.metric("🏆 Best Streak", f"{max_streak} days")
    with col3:
        recent_days = [today_counts.get(today_d - timedelta(days=i), 0) for i in range(7)]
        consistency = len([d for d in recent_days if d >= min_sessions]) / 7 * 100
        st.metric("📊 Weekly Consistency", f"{consistency:.0f}%")

# === NOTES (existing) ===

def add_note(content: str, d: str, user: str):
    nid = hashlib.sha256(f"{d}_{content}_{user}".encode()).hexdigest()
    doc = {"_id": nid, "type":"Note", "date": d, "content": content, "user": user, "created_at": datetime.utcnow()}
    collection_logs.update_one({"_id": nid}, {"$set": doc}, upsert=True)


def render_notes_saver():
    st.header("📝 Daily Notes")
    with st.form("note_form", clear_on_submit=True):
        c1, c2 = st.columns([1,3])
        with c1:
            d = st.date_input("Date", now_ist())
        with c2:
            content = st.text_area("Your thoughts...", height=140)
        if st.form_submit_button("💾 Save Note"):
            if content.strip():
                add_note(content.strip(), d.date().isoformat(), st.session_state.user)
                st.success("Saved")
            else:
                st.warning("Add some content")


def render_notes_viewer():
    st.header("🗂️ Notes Viewer")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("From", now_ist().date()-timedelta(days=7))
    with c2:
        end = st.date_input("To", now_ist().date())
    q = {"type":"Note","user": st.session_state.user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
    notes = list(collection_logs.find(q).sort("date", -1))
    if notes:
        for n in notes:
            st.subheader(f"📅 {n['date']}")
            st.write(n['content'])
            st.divider()
    else:
        st.info("No notes in this range")

# === MAIN ===

def main():
    render_header()
    st.divider()
    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer()
    elif page == "📅 Weekly Planner":
        render_weekly_planner()
    elif page == "🧠 Reflection":
        render_reflection_page()
    elif page == "🧭 Weekly Review":
        render_weekly_review()
    elif page == "📊 Analytics":
        render_analytics_page()
    elif page == "📝 Notes Saver":
        render_notes_saver()
    elif page == "🗂️ Notes Viewer":
        render_notes_viewer()

if __name__ == "__main__":
    main()
