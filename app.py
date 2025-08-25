# app.py
import streamlit as st
import time
from datetime import datetime, timedelta, date
import pandas as pd
import pytz
from pymongo import MongoClient, ASCENDING, DESCENDING
from typing import Dict, List, Optional, Tuple
import math

# =============================
# APP CONFIG
# =============================
st.set_page_config(
    page_title="Focus Timer • Weekly Priorities",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Focus Timer - Weekly Planning & Analytics (v2 schema)"}
)

IST = pytz.timezone("Asia/Kolkata")
POMODORO_MIN = 25
BREAK_MIN = 5
SOUND_PATH = "https://github.com/prashanth-ds-ml/Time_Tracker/raw/refs/heads/main/one_piece_overtake.mp3"

# =============================
# DB INIT
# =============================
@st.cache_resource
def get_db():
    try:
        uri = st.secrets["mongo_uri"]
        client = MongoClient(uri)
        db = client["time_tracker_db"]
        return db
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        st.stop()

db = get_db()
col_plans = db["weekly_plans"]   # registry + plan docs
col_days  = db["user_days"]      # daily rollups + sessions

def ensure_indexes():
    try:
        col_plans.create_index([("user", ASCENDING), ("type", ASCENDING)], name="user_type")
        col_plans.create_index([("_id", ASCENDING)], name="_id_unique", unique=True)
        col_days.create_index([("user", ASCENDING), ("date", ASCENDING)], name="user_date", unique=True)
    except Exception as e:
        st.warning(f"Index ensure notice: {e}")

ensure_indexes()

# =============================
# HELPERS (TIME / MATH)
# =============================
def now_ist() -> datetime:
    return datetime.now(IST)

def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday

def week_bounds_ist(d: date) -> Tuple[date, date]:
    start = monday_of(d)
    return start, start + timedelta(days=6)

def week_day_counts(week_start: date) -> Tuple[int, int]:
    wd = sum(1 for i in range(7) if (week_start + timedelta(days=i)).weekday() < 5)
    we = 7 - wd
    return wd, we

def time_to_minutes(tstr: str) -> Optional[int]:
    try:
        dt = datetime.strptime(tstr, "%I:%M %p")
        return dt.hour * 60 + dt.minute
    except Exception:
        return None

def safe_div(n, d, default=0.0):
    try:
        if d in (None, 0):
            return default
        return float(n) / float(d)
    except Exception:
        return default

def pct_or_dash(n, d, decimals=0):
    if d is None or d <= 0:
        return "—"
    pct = 100.0 * safe_div(n, d, 0.0)
    return f"{pct:.{decimals}f}%"

def gini_from_counts(counts: List[int]):
    arr = [int(c) for c in counts if c is not None and c >= 0]
    if not arr: return 0.0
    arr = sorted(arr)
    n = len(arr); s = sum(arr)
    if s == 0: return 0.0
    cum = 0.0
    for i, x in enumerate(arr, 1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n

def entropy_norm_from_counts(counts: List[int]):
    arr = [int(c) for c in counts if c is not None and c > 0]
    k = len(arr)
    if k <= 1: return 0.0
    s = float(sum(arr))
    H = -sum((c/s) * math.log((c/s), 2) for c in arr)
    return H / math.log(k, 2)

# =============================
# DATA LAYER (v2 schema)
# =============================
DEFAULTS = {
    "weekday_poms": 3,
    "weekend_poms": 5,
    "auto_break": True,
    "custom_categories": ["Learning","Projects","Research","Planning"]
}

@st.cache_data(ttl=120)
def all_users() -> List[str]:
    # Prefer registries
    users = col_plans.distinct("user", {"type": "registry"})
    if not users:
        # Fallback to any plans
        users = col_plans.distinct("user", {"type": "plan"})
    if not users:
        # Fallback to user_days
        users = col_days.distinct("user")
    return sorted(users)

def clamp_priority(w):
    try:
        w = int(w)
    except Exception:
        return 2
    return max(1, min(3, w))

@st.cache_data(ttl=60)
def get_registry(user: str) -> Dict:
    doc = col_plans.find_one({"_id": f"{user}|registry"})
    if doc:
        # sanitize defaults
        ud = doc.get("user_defaults", {}) or {}
        for k, v in DEFAULTS.items():
            if k not in ud:
                ud[k] = v
        doc["user_defaults"] = ud
        goals = doc.get("goals", {}) or {}
        # clamp priority weights
        for gid, g in goals.items():
            if "priority_weight" in g:
                g["priority_weight"] = clamp_priority(g["priority_weight"])
        return doc
    # create registry
    reg = {
        "_id": f"{user}|registry",
        "type": "registry",
        "user": user,
        "user_defaults": DEFAULTS.copy(),
        "goals": {},  # gid -> {title, status, goal_type, priority_weight, target_poms, ...}
        "schema_version": 2,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    col_plans.insert_one(reg)
    get_registry.clear()
    all_users.clear()
    return reg

def update_registry_defaults(user: str, weekday: int, weekend: int):
    col_plans.update_one(
        {"_id": f"{user}|registry"},
        {"$set": {
            "user_defaults.weekday_poms": int(weekday),
            "user_defaults.weekend_poms": int(weekend),
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )
    get_registry.clear()

def upsert_registry_goal(user: str, gid: str, title: str, **fields):
    # merges the goal under registry.goals[gid]
    update = {"goals."+gid+".title": title, "updated_at": datetime.utcnow()}
    for k, v in fields.items():
        if k == "priority_weight":
            v = clamp_priority(v)
        update["goals."+gid+"."+k] = v
    col_plans.update_one(
        {"_id": f"{user}|registry"},
        {"$set": update},
        upsert=True
    )
    get_registry.clear()

@st.cache_data(ttl=60)
def get_plan(user: str, week_start: date, create_if_missing: bool = True) -> Optional[Dict]:
    _id = f"{user}|{week_start.isoformat()}"
    doc = col_plans.find_one({"_id": _id})
    if doc:
        # normalize capacity shape
        cap = doc.get("capacity", {}) or {}
        if "total" not in cap:
            wd, we = week_day_counts(week_start)
            weekday = cap.get("weekday", get_registry(user)["user_defaults"]["weekday_poms"])
            weekend = cap.get("weekend", get_registry(user)["user_defaults"]["weekend_poms"])
            cap["total"] = int(weekday)*wd + int(weekend)*we
            doc["capacity"] = cap
        return doc
    if not create_if_missing:
        return None
    # create empty plan
    reg = get_registry(user)
    weekday = reg["user_defaults"]["weekday_poms"]
    weekend = reg["user_defaults"]["weekend_poms"]
    wd, we = week_day_counts(week_start)
    total = weekday*wd + weekend*we
    plan = {
        "_id": _id,
        "type": "plan",
        "user": user,
        "week_start": week_start.isoformat(),
        "week_end": (week_start + timedelta(days=6)).isoformat(),
        "capacity": {"weekday": int(weekday), "weekend": int(weekend), "total": int(total)},
        "allocations": {},
        "goals": [],
        "goals_embedded": [],
        "stats": {},
        "schema_version": 2,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    col_plans.insert_one(plan)
    get_plan.clear()
    return plan

def save_plan_allocations(user: str, week_start: date, allocations: Dict[str, int]):
    # Snapshot from registry for goals_embedded
    reg = get_registry(user)
    goals = reg.get("goals", {}) or {}
    goals_emb = []
    for gid, planned in allocations.items():
        g = goals.get(gid, {})
        goals_emb.append({
            "goal_id": gid,
            "title": g.get("title", "(missing)"),
            "priority_weight": clamp_priority(g.get("priority_weight", 2)),
            "status_at_plan": g.get("status", "In Progress"),
            "planned": int(planned),
            "carryover_in": int(0),
            "carryover_out": int(0),
        })
    goals_list = list(allocations.keys())
    cap = get_plan(user, week_start, True).get("capacity", {})
    col_plans.update_one(
        {"_id": f"{user}|{week_start.isoformat()}"},
        {"$set": {
            "allocations": {gid: int(v) for gid, v in allocations.items()},
            "goals": goals_list,
            "goals_embedded": goals_emb,
            "capacity": cap,
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )
    get_plan.clear()

@st.cache_data(ttl=60)
def user_days_between(user: str, start: date, end: date) -> List[Dict]:
    return list(col_days.find({"user": user, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}))

def week_actuals_by_goal(user: str, week_start: date) -> Dict[str, int]:
    start, end = week_start, week_start + timedelta(days=6)
    days = user_days_between(user, start, end)
    counts: Dict[str, int] = {}
    for d in days:
        for s in d.get("sessions", []) or []:
            if s.get("t") == "W":
                gid = s.get("gid")
                if gid:
                    counts[gid] = counts.get(gid, 0) + 1
    return counts

# =============================
# ALLOCATION UTILS
# =============================
def proportional_allocation(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    # clamp weights and make sure at least 1
    W = {gid: max(1, clamp_priority(w)) for gid, w in weights.items()}
    total_w = sum(W.values()) if W else 1
    raw = {gid: (w / total_w) * total for gid, w in W.items()}
    base = {gid: int(v) for gid, v in raw.items()}
    diff = total - sum(base.values())
    if diff != 0:
        fracs = sorted(((gid, raw[gid] - int(raw[gid])) for gid in raw), key=lambda x: x[1], reverse=True)
        i = 0
        while diff != 0 and fracs:
            gid = fracs[i % len(fracs)][0]
            base[gid] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1
    return base

# =============================
# TIMER: WRITE TO user_days
# =============================
def sound_alert():
    st.components.v1.html(
        f"""
        <audio id="beep" autoplay>
            <source src="{SOUND_PATH}" type="audio/mpeg">
        </audio>
        <script>
            const a = document.getElementById('beep');
            if (a) {{ a.volume=0.6; a.play().catch(()=>{{}}); }}
        </script>
        """, height=0
    )

def append_session_to_user_day(user: str, is_break: bool, dur_min: int, gid: Optional[str], cat: Optional[str], task: str):
    today = now_ist().date().isoformat()
    tstamp = now_ist().strftime("%I:%M %p")
    s = {"t": "B" if is_break else "W", "dur": int(dur_min), "time": tstamp}
    if not is_break:
        if gid: s["gid"] = gid
        if cat: s["cat"] = cat
        if task: s["task"] = task
    # upsert day
    doc = col_days.find_one({"user": user, "date": today})
    if not doc:
        doc = {
            "_id": f"{user}|{today}",
            "user": user,
            "date": today,
            "sessions": [s],
            "schema_version": 2,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        col_days.insert_one(doc)
    else:
        col_days.update_one({"_id": doc["_id"]}, {"$push": {"sessions": s}, "$set": {"updated_at": datetime.utcnow()}})
    user_days_between.clear()

# =============================
# SESSION STATE
# =============================
def init_state():
    defaults = {
        "user": None,
        "page": "🎯 Focus Timer",
        "planning_week_date": now_ist().date(),
        "start_time": None,
        "is_break": False,
        "active_gid": None,
        "active_goal_title": "",
        "task": "",
        "show_on_hold": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_timer_state():
    st.session_state.start_time = None
    st.session_state.is_break = False
    st.session_state.active_gid = None
    st.session_state.active_goal_title = ""
    st.session_state.task = ""

init_state()

# =============================
# UI: WEEKLY PLANNER
# =============================
def ui_last_week_actions(user: str, this_week_start: date):
    st.subheader("Last Week Recap & Actions")
    prev_start = this_week_start - timedelta(days=7)
    prev_plan = get_plan(user, prev_start, create_if_missing=False)

    if not prev_plan or not (prev_plan.get("allocations") or {}):
        st.info("No previous plan found or it had no allocations.")
        return {}

    # Compute actuals from user_days
    actual_prev = week_actuals_by_goal(user, prev_start)
    alloc_prev = {gid: int(v) for gid, v in (prev_plan.get("allocations") or {}).items()}

    rows = []
    for gid, planned in alloc_prev.items():
        actual = int(actual_prev.get(gid, 0))
        carry = max(0, planned - actual)
        title = "(missing)"
        # Try goals_embedded title
        for ge in (prev_plan.get("goals_embedded") or []):
            if ge.get("goal_id") == gid:
                title = ge.get("title", title)
                break
        rows.append({"goal_id": gid, "title": title, "planned": planned, "actual": actual, "carry": carry})

    df = pd.DataFrame(rows).sort_values("title")
    st.dataframe(df.rename(columns={"goal_id":"Goal ID","title":"Goal","planned":"Planned","actual":"Actual","carry":"Carry"}),
                 use_container_width=True, hide_index=True)

    actions = {}
    st.caption("Choose an action per goal (default: Continue)")
    for r in rows:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{r['title']}**")
        with col2:
            actions[r["goal_id"]] = st.selectbox(
                "Action", ["Continue", "Rollover carry", "Hold", "Complete"],
                key=f"act_{r['goal_id']}_{prev_start}"
            )
    st.divider()
    return {"rows": rows, "actions": actions, "prev_start": prev_start}

def ui_this_week_planner(user: str, week_start: date):
    st.subheader("This Week Plan")

    reg = get_registry(user)
    plan = get_plan(user, week_start, create_if_missing=True)

    # Capacity controls
    weekday = st.number_input("Weekday avg", 0, 12, value=int(reg["user_defaults"]["weekday_poms"]))
    weekend = st.number_input("Weekend avg", 0, 12, value=int(reg["user_defaults"]["weekend_poms"]))
    wd, we = week_day_counts(week_start)
    total_cap = int(weekday)*wd + int(weekend)*we
    st.metric(f"Capacity {week_start} → {week_start + timedelta(days=6)}", f"{total_cap}")

    if (weekday != reg["user_defaults"]["weekday_poms"]) or (weekend != reg["user_defaults"]["weekend_poms"]):
        if st.button("💾 Save Capacity Defaults"):
            update_registry_defaults(user, int(weekday), int(weekend))
            st.success("Defaults updated.")
            st.rerun()

    st.markdown("---")

    # Goal list to plan: from registry (New, In Progress) + optional On Hold
    all_goals: Dict[str, Dict] = reg.get("goals", {}) or {}
    def status_of(g): return (g or {}).get("status", "In Progress")
    candidates = {
        gid: g for gid, g in all_goals.items()
        if status_of(g) in (["New","In Progress"] + (["On Hold"] if st.session_state.show_on_hold else []))
    }

    st.checkbox("Include On Hold goals", value=st.session_state.show_on_hold, key="show_on_hold")

    if not candidates:
        with st.expander("➕ Add Goal"):
            g_title = st.text_input("Title")
            g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0)
            g_weight = st.select_slider("Priority", options=[1,2,3], value=2)
            if st.button("Add"):
                gid = f"{hash(user+g_title) & 0xffffffff:08x}"  # quick deterministic-ish id
                upsert_registry_goal(user, gid, g_title.strip(), status="In Progress", goal_type=g_type, priority_weight=int(g_weight))
                st.success("Goal added.")
                st.rerun()
        st.info("No active goals to plan. Add some above.")
        return

    # Editable priorities
    st.markdown("### Priority Weights")
    weights = {}
    cols = st.columns(min(4, max(1, len(candidates))))
    for i, (gid, g) in enumerate(sorted(candidates.items(), key=lambda x: x[1].get("title",""))):
        with cols[i % len(cols)]:
            st.write(f"**{g.get('title','(missing)')}**")
            w = st.select_slider("Priority", options=[1,2,3], value=clamp_priority(g.get("priority_weight", 2)), key=f"w_{gid}")
            weights[gid] = int(w)

    if st.button("💾 Update Priorities"):
        for gid, w in weights.items():
            upsert_registry_goal(user, gid, candidates[gid].get("title","(missing)"), priority_weight=int(w))
        st.success("Priorities updated.")
        st.rerun()

    st.markdown("---")

    # Auto allocation (from priorities) vs current plan allocations
    st.markdown("### Allocate Weekly Pomodoros")
    auto = proportional_allocation(total_cap, weights)
    current_alloc = {gid: int(v) for gid, v in (plan.get("allocations") or {}).items()}
    edited: Dict[str, int] = {}

    cols2 = st.columns(min(4, max(1, len(candidates))))
    for i, (gid, g) in enumerate(sorted(candidates.items(), key=lambda x: x[1].get("title",""))):
        with cols2[i % len(cols2)]:
            base = current_alloc.get(gid, auto.get(gid, 0))
            val = st.number_input(
                f"{g.get('title','(missing)')}",
                min_value=0, max_value=total_cap, value=int(base), step=1, key=f"a_{gid}_{week_start}"
            )
            edited[gid] = int(val)

    ssum = sum(edited.values())
    if ssum != total_cap:
        st.warning(f"Allocations sum to {ssum}, not {total_cap}.")
        if st.button("🔁 Normalize to capacity"):
            edited = proportional_allocation(total_cap, {gid: max(1, v) for gid, v in edited.items()})
            for gid, v in edited.items():
                st.session_state[f"a_{gid}_{week_start}"] = v
            st.rerun()

    if st.button("📌 Save / Update This Week Plan", type="primary"):
        # store capacity into plan before allocations
        col_plans.update_one(
            {"_id": f"{user}|{week_start.isoformat()}"},
            {"$set": {"capacity": {"weekday": int(weekday), "weekend": int(weekend), "total": int(total_cap)},
                      "updated_at": datetime.utcnow()}},
            upsert=True
        )
        save_plan_allocations(user, week_start, edited)
        st.success("Weekly plan saved.")
        st.rerun()

def ui_apply_last_week_actions(payload: Dict, user: str, this_week_start: date):
    if not payload:
        return
    rows = payload["rows"]; actions = payload["actions"]
    plan = get_plan(user, this_week_start, create_if_missing=True)
    current_alloc = {gid: int(v) for gid, v in (plan.get("allocations") or {}).items()}

    if st.button("✅ Apply Actions to Current Week"):
        reg = get_registry(user)
        goals_map = reg.get("goals", {}) or {}

        # Merge changes
        for r in rows:
            gid = r["goal_id"]
            act = actions.get(gid, "Continue")
            if act == "Rollover carry" and r["carry"] > 0:
                current_alloc[gid] = int(current_alloc.get(gid, 0)) + int(r["carry"])
            elif act == "Hold":
                # move to On Hold
                upsert_registry_goal(user, gid, goals_map.get(gid, {}).get("title", r["title"]), status="On Hold")
                # optionally remove from this week's allocations
                if gid in current_alloc: current_alloc.pop(gid, None)
            elif act == "Complete":
                upsert_registry_goal(user, gid, goals_map.get(gid, {}).get("title", r["title"]), status="Completed")
                if gid in current_alloc: current_alloc.pop(gid, None)
            # Continue = no changes

        save_plan_allocations(user, this_week_start, current_alloc)
        st.success("Applied. This week's plan updated.")
        st.rerun()

def render_weekly_planner(user: str, week_pick: date):
    st.header("📅 Weekly Planner")

    week_start, _ = week_bounds_ist(week_pick)

    colL, colR = st.columns([1, 1])
    with colL:
        st.markdown("#### Week Selector")
        d = st.date_input("Week of", value=week_pick)
        if d != week_pick:
            st.session_state.planning_week_date = d
            st.rerun()

    # LEFT: Last Week Recap & Actions
    col1, col2 = st.columns([1, 1])
    with col1:
        payload = ui_last_week_actions(user, week_start)
        ui_apply_last_week_actions(payload, user, week_start)

    # RIGHT: This Week Plan
    with col2:
        ui_this_week_planner(user, week_start)

# =============================
# FOCUS TIMER (robust for new users)
# =============================
def render_focus_timer(user: str):
    st.header("🎯 Focus Timer")

    reg = get_registry(user)

    # Auto-break toggle (from registry defaults)
    auto_break = st.toggle("Auto-start break (5m)", value=bool(reg["user_defaults"].get("auto_break", True)))
    if auto_break != reg["user_defaults"].get("auto_break", True):
        col_plans.update_one(
            {"_id": f"{user}|registry"},
            {"$set": {"user_defaults.auto_break": bool(auto_break), "updated_at": datetime.utcnow()}}
        )
        get_registry.clear()

    # If timer running, show and tick
    if st.session_state.start_time:
        duration = BREAK_MIN*60 if st.session_state.is_break else POMODORO_MIN*60
        remaining = int(st.session_state.start_time + duration - time.time())
        if remaining > 0:
            mins, secs = divmod(remaining, 60)
            label = "Break" if st.session_state.is_break else f"Working on: {st.session_state.task or '(no task)'}"
            st.subheader(f"{'🧘' if st.session_state.is_break else '💼'} {label}")
            _, c, _ = st.columns([1,2,1])
            with c:
                st.markdown(f"<h1 style='text-align:center;font-size:4rem;'>⏱️ {mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
            st.progress(1 - (remaining / duration))
            time.sleep(1); st.rerun(); return
        else:
            # Save session and play sound before break auto-start
            was_break = st.session_state.is_break
            append_session_to_user_day(
                user=user,
                is_break=was_break,
                dur_min=(BREAK_MIN if was_break else POMODORO_MIN),
                gid=(None if was_break else st.session_state.active_gid),
                cat=(None if was_break else st.session_state.active_goal_title),
                task=(st.session_state.task if not was_break else "")
            )
            sound_alert()
            st.success("🎉 Session complete!")
            st.balloons()
            # Reset
            reset_timer_state()
            if (not was_break) and auto_break:
                st.toast("☕ Auto-starting a 5-minute break")
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.rerun()
            return

    # No active timer → setup
    st.subheader("This Week at a Glance")
    week_start, week_end = week_bounds_ist(now_ist().date())
    plan = get_plan(user, week_start, create_if_missing=True)
    alloc = plan.get("allocations") or {}
    goals_emb = plan.get("goals_embedded") or []

    if not alloc:
        st.info("No weekly plan saved yet. Go to **Weekly Planner** to create allocations.")
    else:
        # Show simple per-goal progress for this week
        actuals = week_actuals_by_goal(user, week_start)
        rows = []
        for ge in goals_emb:
            gid = ge.get("goal_id")
            title = ge.get("title","(missing)")
            planned = int(alloc.get(gid, 0))
            actual = int(actuals.get(gid, 0))
            rows.append({"Goal": title, "Progress": f"{actual}/{planned}"})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True, height=min(280, 38*len(df)+38))

    st.markdown("---")

    # Start a session (Weekly Goal or Custom)
    mode = st.radio("Mode", ["Weekly Goal", "Custom (Unplanned)"], horizontal=True)
    if mode == "Weekly Goal":
        titles = [(ge.get("title","(missing)"), ge.get("goal_id")) for ge in goals_emb]
        if not titles:
            st.warning("No active goals in this week's plan.")
            selected_gid = None; selected_title = ""
        else:
            labels = [t for (t, _) in titles]
            sel_idx = st.selectbox("Goal", options=range(len(labels)), format_func=lambda i: labels[i])
            selected_gid = titles[sel_idx][1]
            selected_title = titles[sel_idx][0]

        task = st.text_input("Task (micro-task)")
        st.session_state.active_gid = selected_gid
        st.session_state.active_goal_title = selected_title
        st.session_state.task = task

        colA, colB = st.columns(2)
        with colA:
            disabled = (not selected_gid) or (not task.strip())
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colB:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.rerun()

    else:
        # Custom category (not tied to gid)
        cats = reg["user_defaults"].get("custom_categories", DEFAULTS["custom_categories"])
        cat_sel = st.selectbox("📂 Category", cats)
        task = st.text_input("Task (micro-task)")
        st.session_state.active_gid = None
        st.session_state.active_goal_title = cat_sel
        st.session_state.task = task

        colA, colB = st.columns(2)
        with colA:
            disabled = not (cat_sel and task.strip())
            if st.button("▶️ Start Work (25m)", type="primary", use_container_width=True, disabled=disabled):
                st.session_state.start_time = time.time()
                st.session_state.is_break = False
                st.rerun()
        with colB:
            if st.button("☕ Break (5m)", use_container_width=True):
                st.session_state.start_time = time.time()
                st.session_state.is_break = True
                st.rerun()

# =============================
# ANALYTICS
# =============================
def render_analytics(user: str):
    st.header("📊 Analytics & Review")

    # Mode
    try:
        mode = st.segmented_control("Mode", options=["Week Review", "Trends"], default="Week Review", key="analytics_mode")
    except Exception:
        mode = st.radio("Mode", ["Week Review", "Trends"], horizontal=True, index=0)

    if mode == "Week Review":
        pick = st.date_input("Review week of", value=monday_of(now_ist().date()))
        wk_start, wk_end = week_bounds_ist(pick)
        plan = get_plan(user, wk_start, create_if_missing=False)
        if not plan:
            st.info("No plan found for that week.")
            return

        # Actuals from user_days
        actual_by_gid = week_actuals_by_goal(user, wk_start)
        total_actual = sum(actual_by_gid.values())
        planned_alloc = {gid: int(v) for gid, v in (plan.get("allocations") or {}).items()}
        total_planned = sum(planned_alloc.values())

        # Build per-goal table and chart
        title_map = {ge.get("goal_id"): ge.get("title","(missing)") for ge in (plan.get("goals_embedded") or [])}
        rows = []
        for gid, planned in planned_alloc.items():
            rows.append({
                "Goal": title_map.get(gid, "(missing)"),
                "Planned": int(planned),
                "Actual": int(actual_by_gid.get(gid, 0))
            })
        df = pd.DataFrame(rows).sort_values("Goal")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Plan Adherence", pct_or_dash(total_actual, total_planned))
        with c2:
            st.metric("Capacity Utilization", pct_or_dash(total_actual, total_planned))
        # Deep-work proxy: sessions of 25 min are all "deep" in this model; we don't store durations per session in plan view.
        # Read from user_days (>=23 min work)
        deep = 0; all_work = 0
        for d in user_days_between(user, wk_start, wk_end):
            for s in d.get("sessions", []) or []:
                if s.get("t") == "W":
                    all_work += 1
                    if int(s.get("dur", 0)) >= 23:
                        deep += 1
        with c3:
            st.metric("Deep-work %", pct_or_dash(deep, all_work))
        with c4:
            # entropy across goal actuals
            st.metric("Balance (Entropy)", f"{entropy_norm_from_counts(list(actual_by_gid.values())):.2f}")

        st.divider()
        st.subheader("Planned vs Actual (Per Goal)")
        if not df.empty:
            st.bar_chart(df.set_index("Goal")[["Planned","Actual"]], height=300, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True, height=min(300, 38*len(df)+38))
        else:
            st.info("No goal allocations in this plan.")

        st.markdown("### Run-rate vs Expected (Goals only)")
        if total_planned > 0:
            days = [wk_start + timedelta(days=i) for i in range(7)]
            actual_cum = []
            expected_cum = []
            running = 0
            for i, d in enumerate(days, 1):
                # count work sessions up to this day
                upto = [x for x in user_days_between(user, wk_start, d)]
                running = sum(
                    1 for dd in upto for s in dd.get("sessions", []) or [] if s.get("t") == "W" and s.get("gid")
                )
                actual_cum.append(running)
                expected_cum.append(int(round(total_planned * (i/7.0))))
            rr = pd.DataFrame({"day":[d.strftime("%a %d") for d in days], "Expected":expected_cum, "Actual":actual_cum}).set_index("day")
            st.line_chart(rr, height=260, use_container_width=True)

        st.markdown("### Start-time Stability")
        # median start time mins per day
        series = []
        for d in user_days_between(user, wk_start, wk_end):
            starts = []
            for s in d.get("sessions", []) or []:
                if s.get("t") == "W" and isinstance(s.get("time"), str):
                    m = time_to_minutes(s["time"])
                    if m is not None: starts.append(m)
            if starts:
                series.append({"date": d["date"], "median_start": int(pd.Series(starts).median())})
        if series:
            df_st = pd.DataFrame(series).sort_values("date").set_index("date")
            st.line_chart(df_st, height=220, use_container_width=True)
        else:
            st.info("No start-time data this week.")

        st.markdown("### Work vs Break Minutes (Daily)")
        bars = []
        for d in user_days_between(user, wk_start, wk_end):
            w = sum(int(s.get("dur",0)) for s in d.get("sessions", []) or [] if s.get("t")=="W")
            b = sum(int(s.get("dur",0)) for s in d.get("sessions", []) or [] if s.get("t")=="B")
            bars.append({"date": d["date"][5:], "Work": w, "Break": b})
        if bars:
            dfwb = pd.DataFrame(bars).set_index("date")
            st.bar_chart(dfwb, height=260, use_container_width=True)

    else:
        # Trends across last 30 days
        today = now_ist().date()
        start = today - timedelta(days=29)
        days = user_days_between(user, start, today)
        if not days:
            st.info("No data in the last 30 days.")
            return

        # Daily focus minutes (work)
        daily = []
        for d in sorted(days, key=lambda x: x["date"]):
            w = sum(int(s.get("dur",0)) for s in d.get("sessions", []) or [] if s.get("t")=="W")
            daily.append({"date": d["date"][5:], "minutes": w})
        if sum(x["minutes"] for x in daily) > 0:
            st.subheader("📈 Daily Focus Minutes (Last 30 days)")
            st.bar_chart(pd.DataFrame(daily).set_index("date"), height=300, use_container_width=True)

        # Category minutes (from user_days.by_category_minutes)
        cat_totals: Dict[str, int] = {}
        for d in days:
            for cat, mins in (d.get("by_category_minutes") or {}).items():
                cat_totals[cat] = cat_totals.get(cat, 0) + int(mins)
        if cat_totals:
            st.subheader("📊 Time by Category")
            df_cat = pd.DataFrame([{"Category": k, "Minutes": v} for k, v in cat_totals.items()]).sort_values("Minutes", ascending=False)
            st.bar_chart(df_cat.set_index("Category"), height=300, use_container_width=True)
            st.dataframe(df_cat.assign(Hours=(df_cat["Minutes"]/60).round(1)),
                         use_container_width=True, hide_index=True, height=min(300, 38*len(df_cat)+38))

        # Start-time distribution
        starts = []
        for d in days:
            for s in d.get("sessions", []) or []:
                if s.get("t") == "W" and isinstance(s.get("time"), str):
                    m = time_to_minutes(s["time"])
                    if m is not None: starts.append(m // 60)
        if starts:
            st.subheader("🕘 Start-hour Distribution (Last 30 days)")
            sh = pd.Series(starts, name="count").value_counts().sort_index()
            df_sh = pd.DataFrame({"hour": [f"{(h%12) or 12}{'AM' if h<12 else 'PM'}" for h in sh.index], "count": sh.values}).set_index("hour")
            st.bar_chart(df_sh, height=240, use_container_width=True)

        # Simple insights
        st.subheader("🔍 Insights")
        best_day = max(daily, key=lambda x: x["minutes"], default=None)
        if best_day:
            st.metric("Best day (mins)", best_day["minutes"], best_day["date"])
        total_w = sum(x["minutes"] for x in daily)
        if cat_totals and total_w > 0:
            top_cat, top_min = max(cat_totals.items(), key=lambda x: x[1])
            st.metric("Top category share", f"{(top_min/total_w)*100:.0f}%")
        # Break hygiene
        total_work_blocks = sum(1 for d in days for s in d.get("sessions", []) or [] if s.get("t")=="W")
        total_break_blocks = sum(1 for d in days for s in d.get("sessions", []) or [] if s.get("t")=="B")
        st.metric("Break skip / extend", f"{pct_or_dash(max(0,total_work_blocks-total_break_blocks), total_work_blocks)} / {pct_or_dash(max(0,total_break_blocks-total_work_blocks), total_work_blocks)}")

# =============================
# JOURNAL (lightweight, uses user_days.notes)
# =============================
def render_journal(user: str):
    st.header("🧾 Journal")
    tabs = st.tabs(["Add Note", "Browse Notes (Last 14 days)"])
    with tabs[0]:
        d = st.date_input("Date", value=now_ist().date())
        content = st.text_area("Your note", height=140)
        if st.button("💾 Save Note"):
            if content.strip():
                doc = col_days.find_one({"user": user, "date": d.isoformat()})
                if not doc:
                    doc = {"_id": f"{user}|{d.isoformat()}", "user": user, "date": d.isoformat(),
                           "sessions": [], "notes": [{"content": content.strip(), "created_at": datetime.utcnow()}],
                           "schema_version": 2, "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()}
                    col_days.insert_one(doc)
                else:
                    col_days.update_one({"_id": doc["_id"]},
                                        {"$push": {"notes": {"content": content.strip(), "created_at": datetime.utcnow()}},
                                         "$set": {"updated_at": datetime.utcnow()}})
                st.success("Saved note.")
                user_days_between.clear()
            else:
                st.warning("Write something first.")
    with tabs[1]:
        start = (now_ist().date() - timedelta(days=13)).isoformat()
        end = now_ist().date().isoformat()
        days = user_days_between(user, date.fromisoformat(start), date.fromisoformat(end))
        days = sorted(days, key=lambda x: x["date"], reverse=True)
        if not days:
            st.info("No notes found.")
        for d in days:
            notes = d.get("notes") or []
            if not notes:
                continue
            st.subheader(f"📅 {d['date']}")
            for n in notes:
                st.write("• " + n.get("content","(empty)"))
            st.divider()

# =============================
# HEADER / ROUTER
# =============================
def main_header_and_router():
    # Users
    users = all_users()
    if not users:
        # bootstrap a default user registry
        _ = get_registry("prashanth")
        users = all_users()

    if st.session_state.user is None or st.session_state.user not in users:
        st.session_state.user = users[0]

    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        sel = st.selectbox("👤 User", users, index=users.index(st.session_state.user))
        if sel != st.session_state.user:
            st.session_state.user = sel
            reset_timer_state()
            all_users.clear()
            get_registry.clear()
            get_plan.clear()
            user_days_between.clear()
            st.rerun()
    with c2:
        pages = ["🎯 Focus Timer", "📅 Weekly Planner", "📊 Analytics & Review", "🧾 Journal"]
        st.session_state.page = st.selectbox("📍 Navigate", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    with c3:
        with st.expander("➕ Quick Add Goal"):
            g_title = st.text_input("Title", key="quick_goal_title")
            g_type = st.selectbox("Type", ["Certification","Portfolio","Job Prep","Research","Startup","Learning","Other"], index=0, key="quick_goal_type")
            g_weight = st.select_slider("Priority", options=[1,2,3], value=2, key="quick_goal_weight")
            if st.button("Add Goal", key="quick_goal_add"):
                gid = f"{hash(st.session_state.user+g_title) & 0xffffffff:08x}"
                upsert_registry_goal(st.session_state.user, gid, g_title.strip(), status="In Progress", goal_type=g_type, priority_weight=int(g_weight))
                st.success("Goal added.")
                st.rerun()

    st.divider()

    # Route
    page = st.session_state.page
    if page == "🎯 Focus Timer":
        render_focus_timer(st.session_state.user)
    elif page == "📅 Weekly Planner":
        render_weekly_planner(st.session_state.user, st.session_state.planning_week_date)
    elif page == "📊 Analytics & Review":
        render_analytics(st.session_state.user)
    elif page == "🧾 Journal":
        render_journal(st.session_state.user)

if __name__ == "__main__":
    main_header_and_router()
